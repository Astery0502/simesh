# cython: boundscheck=False, wraparound=False, cdivision=True
"""Batch direct exact-phase actions after provider geometry/slot admission."""
from libc.stdint cimport int64_t, uint8_t


cpdef void apply_direct_actions(
    double[:, :, :, :, ::1] payload,
    int64_t primary_count,
    const uint8_t[:, ::1] kinds,
    const uint8_t[:, ::1] masks,
    const uint8_t[:, ::1] counts,
    const int64_t[:, :, ::1] slots,
    const uint8_t[:, :, ::1] phases,
    const int64_t[:, ::1] directions,
    const int64_t[:, ::1] target_lower,
    const int64_t[:, ::1] target_upper,
    const int64_t[::1] lower,
    const int64_t[::1] upper,
):
    # SAME/FINER read only loaded interiors, so these disjoint destinations
    # can precede COARSER support construction and final physical widening.
    cdef int64_t p, r, a, n, slot, field, i, j, k, x, y, z, width, half
    cdef int64_t lo[3]
    cdef int64_t hi[3]
    cdef int64_t src[3]
    cdef double total
    with nogil:
        for p in range(primary_count):
            for r in range(kinds.shape[1]):
                if masks[p,r] != 0 or (kinds[p,r] != 3 and kinds[p,r] != 4):
                    continue
                if (target_lower[r,0] >= target_upper[r,0] or
                    target_lower[r,1] >= target_upper[r,1] or
                    target_lower[r,2] >= target_upper[r,2]):
                    continue
                for n in range(counts[p,r]):
                    slot = slots[p,r,n]
                    for a in range(3):
                        lo[a] = target_lower[r,a]
                        hi[a] = target_upper[r,a]
                        if kinds[p,r] == 3:
                            src[a] = lo[a] - directions[r,a]*(upper[a]-lower[a])
                        elif directions[r,a] == 0:
                            # FRP-001 exact phase placement, without coordinates.
                            half = (upper[a]-lower[a]) // 2
                            lo[a] = lower[a] + ((phases[p,r,n] >> a) & 1)*half
                            hi[a] = lo[a] + half
                            src[a] = lower[a]
                        elif directions[r,a] < 0:
                            width = hi[a]-lo[a]
                            src[a] = upper[a]-2*width
                        else:
                            src[a] = lower[a]
                    for field in range(payload.shape[1]):
                        for i in range(lo[0],hi[0]):
                            for j in range(lo[1],hi[1]):
                                for k in range(lo[2],hi[2]):
                                    if kinds[p,r] == 3:
                                        payload[p,field,i,j,k] = payload[slot,field,
                                            src[0]+i-lo[0],src[1]+j-lo[1],src[2]+k-lo[2]]
                                    else:
                                        x = src[0]+2*(i-lo[0])
                                        y = src[1]+2*(j-lo[1])
                                        z = src[2]+2*(k-lo[2])
                                        # RST-001 accumulation order is part of the contract.
                                        total = payload[slot,field,x,y,z]
                                        total = total + payload[slot,field,x+1,y,z]
                                        total = total + payload[slot,field,x,y+1,z]
                                        total = total + payload[slot,field,x+1,y+1,z]
                                        total = total + payload[slot,field,x,y,z+1]
                                        total = total + payload[slot,field,x+1,y,z+1]
                                        total = total + payload[slot,field,x,y+1,z+1]
                                        total = total + payload[slot,field,x+1,y+1,z+1]
                                        payload[p,field,i,j,k] = total * 0.125
