import os
import random

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun, per_token_cast_back


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def wait_or_hook(rank, label, event, hook, return_hook):
    if return_hook:
        log(rank, f"calling {label} hook")
        hook()
        log(rank, f"{label} hook returned")
    else:
        log(rank, f"waiting {label} event")
        event.current_stream_wait()
        torch.cuda.synchronize()
        log(rank, f"{label} event wait returned")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))

    num_tokens = int(os.environ.get("LL_DEBUG_TOKENS", "16"))
    hidden = int(os.environ.get("LL_DEBUG_HIDDEN", "7168"))
    num_topk = int(os.environ.get("LL_DEBUG_TOPK", "4"))
    num_experts = int(os.environ.get("LL_DEBUG_EXPERTS", str(max(8, num_ranks * 8))))
    use_fp8 = bool(int(os.environ.get("LL_DEBUG_FP8", "0")))
    round_scale = bool(int(os.environ.get("LL_DEBUG_ROUND_SCALE", "0")))
    use_ue8m0 = bool(int(os.environ.get("LL_DEBUG_UE8M0", "0")))
    return_hook = bool(int(os.environ.get("LL_DEBUG_HOOK", "0")))
    do_combine = bool(int(os.environ.get("LL_DEBUG_COMBINE", "0")))
    zero_copy = bool(int(os.environ.get("LL_DEBUG_ZERO_COPY", "0")))
    dispatch_reps = int(os.environ.get("LL_DEBUG_DISPATCH_REPS", "1"))
    sequence = bool(int(os.environ.get("LL_DEBUG_SEQUENCE", "0")))

    torch.manual_seed(rank)
    random.seed(rank)
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    log(rank, f"creating Buffer rdma={num_rdma_bytes} experts={num_experts}")
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=not bool(int(os.environ.get("LL_DEBUG_DISABLE_NVLINK", "0"))),
        explicitly_destroy=True,
    )
    log(rank, "Buffer created")

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - 128)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    stats = torch.zeros((num_experts // num_ranks,), dtype=torch.int, device="cuda")

    dist.barrier(group)

    if sequence:
        cases = [
            (False, False, False, False),
            (True, False, False, False),
            (True, True, False, False),
            (True, True, True, False),
            (False, False, False, True),
            (True, False, False, True),
        ]
        for case_idx, (case_fp8, case_round, case_ue8m0, case_hook) in enumerate(cases):
            log(
                rank,
                "sequence case "
                f"{case_idx} fp8={case_fp8} round_scale={case_round} "
                f"use_ue8m0={case_ue8m0} hook={case_hook}",
            )
            stats.zero_()
            recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts,
                use_fp8=case_fp8,
                round_scale=case_round,
                use_ue8m0=case_ue8m0,
                cumulative_local_expert_recv_stats=stats,
                async_finish=not case_hook,
                return_recv_hook=case_hook,
            )
            log(rank, f"sequence case {case_idx} dispatch returned")
            wait_or_hook(rank, f"sequence case {case_idx} dispatch", event, hook, case_hook)
            log(rank, f"sequence case {case_idx} recv_count_sum={int(recv_count.sum().item())}")

            if case_fp8:
                simulated_gemm_x = per_token_cast_back(
                    recv_x[0].view(-1, hidden),
                    recv_x[1].contiguous().view(-1, hidden // 128),
                ).view(recv_x[0].shape)
            else:
                simulated_gemm_x = recv_x.clone()

            for case_zero_copy in (False, True):
                if case_zero_copy:
                    buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = (
                        simulated_gemm_x
                    )
                log(
                    rank,
                    f"sequence case {case_idx} combine zero_copy={case_zero_copy}",
                )
                combined_x, event, hook = buffer.low_latency_combine(
                    simulated_gemm_x,
                    topk_idx,
                    topk_weights,
                    handle,
                    async_finish=not case_hook,
                    zero_copy=case_zero_copy,
                    return_recv_hook=case_hook,
                )
                log(rank, f"sequence case {case_idx} combine returned zero_copy={case_zero_copy}")
                wait_or_hook(
                    rank,
                    f"sequence case {case_idx} combine zero_copy={case_zero_copy}",
                    event,
                    hook,
                    case_hook,
                )
                log(
                    rank,
                    "sequence case "
                    f"{case_idx} combine zero_copy={case_zero_copy} "
                    f"sum={float(combined_x.float().sum().item())}",
                )

        buffer.destroy()
        dist.barrier(group)
        dist.destroy_process_group()
        log(rank, "done")
        return

    recv_x = recv_count = handle = None
    for rep in range(dispatch_reps):
        log(
            rank,
            "calling low_latency_dispatch "
            f"rep={rep + 1}/{dispatch_reps} fp8={use_fp8} "
            f"round_scale={round_scale} use_ue8m0={use_ue8m0} "
            f"hook={return_hook}",
        )
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=use_fp8,
            round_scale=round_scale,
            use_ue8m0=use_ue8m0,
            cumulative_local_expert_recv_stats=stats,
            async_finish=not return_hook,
            return_recv_hook=return_hook,
        )
        log(rank, f"low_latency_dispatch returned rep={rep + 1}")
        wait_or_hook(rank, f"dispatch rep={rep + 1}", event, hook, return_hook)

    log(rank, f"recv_count_sum={int(recv_count.sum().item())}")
    if do_combine:
        if use_fp8:
            simulated_gemm_x = per_token_cast_back(
                recv_x[0].view(-1, hidden),
                recv_x[1].contiguous().view(-1, hidden // 128),
            ).view(recv_x[0].shape)
        else:
            simulated_gemm_x = recv_x.clone()

        if zero_copy:
            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = (
                simulated_gemm_x
            )

        log(rank, f"calling low_latency_combine zero_copy={zero_copy}")
        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            async_finish=not return_hook,
            zero_copy=zero_copy,
            return_recv_hook=return_hook,
        )
        log(rank, "low_latency_combine returned")
        wait_or_hook(rank, "combine", combine_event, combine_hook, return_hook)
        log(rank, f"combine result sum={float(combined_x.float().sum().item())}")

    buffer.destroy()
    dist.barrier(group)
    dist.destroy_process_group()
    log(rank, "done")


if __name__ == "__main__":
    main()
