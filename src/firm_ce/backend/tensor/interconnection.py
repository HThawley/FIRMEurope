import numpy as np
from numba import njit

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def Interconnection(
    solution,
    Fillt,
    Surplust,
    netflowt,
    Importt,
    Exportt
) -> None:  # noqa: C901
    nodes = solution.static.nodes
    nhvi = solution.static.nhvi

    cap_fwd = solution.operations.cap_fwd
    cap_rev = solution.operations.cap_rev
    eff_fwd = solution.operations.eff_fwd
    eff_rev = solution.operations.eff_rev
    eff = solution.operations.eff
    visited = solution.operations.visited
    parent_node = solution.operations.parent_node
    parent_line = solution.operations.parent_line
    path_nodes = solution.operations.path_nodes
    path_lines = solution.operations.path_lines

    # Sort nodes descending by Fill
    priority_order = np.argsort(Fillt)[::-1]
    total_surplus = Surplust.sum()

    # map out network
    for line in range(nhvi):
        line_eff = solution.static.line_efficiencies[line]

        # Forward direction (start -> end)
        if netflowt[line] < -TOLERANCE:  # Countering existing reverse flow
            cap_fwd[line] = -netflowt[line] * line_eff
            # Eff > 1.0 because injecting 1 MW here saves >1 MW at the original source
            eff_fwd[line] = 1.0 / line_eff
        else:  # Pushing new forward flow
            cap_fwd[line] = solution.assets.Clines[line] - netflowt[line]
            eff_fwd[line] = line_eff

        # Reverse direction (end -> start)
        if netflowt[line] > TOLERANCE:  # Countering existing forward flow
            cap_rev[line] = netflowt[line] * line_eff
            eff_rev[line] = 1.0 / line_eff
        else:  # Pushing new reverse flow
            cap_rev[line] = solution.assets.Clines[line] + netflowt[line]
            eff_rev[line] = line_eff

    for n in priority_order:
        if Fillt[n] < TOLERANCE:
            break  # No more significant deficits

        while Fillt[n] > TOLERANCE and total_surplus > TOLERANCE:

            # Dijkstra setup
            eff.fill(0.0)
            eff[n] = 1.0
            visited.fill(False)
            parent_node.fill(-1)
            parent_line.fill(-1)
            best_surplus_node = -1

            # Dijkstra execution
            for _ in range(nodes):
                curr = -1
                max_e = -1.0
                for i in range(nodes):
                    if not visited[i] and eff[i] > max_e:
                        max_e = eff[i]
                        curr = i

                if curr == -1 or max_e == 0.0:
                    break  # Unreachable or fully explored

                if Surplust[curr] > TOLERANCE:
                    # early exit
                    best_surplus_node = curr
                    break

                visited[curr] = True
                pdonors_arr = solution.static.cache_0_donors[curr]

                if pdonors_arr.shape[1] == 0:
                    continue

                neighbors = pdonors_arr[0]
                lines = pdonors_arr[1]

                for idx in range(len(neighbors)):
                    nxt = neighbors[idx]
                    line = lines[idx]

                    if visited[nxt]:
                        continue

                    # Physical flow is nxt -> curr
                    is_fwd = (nxt == solution.static.network[line, 0])
                    avail_cap = cap_fwd[line] if is_fwd else cap_rev[line]
                    edge_e = eff_fwd[line] if is_fwd else eff_rev[line]

                    if avail_cap < TOLERANCE:
                        continue  # Congested or hit zero-crossing limit

                    new_e = eff[curr] * edge_e
                    if new_e > eff[nxt]:
                        eff[nxt] = new_e
                        parent_node[nxt] = curr
                        parent_line[nxt] = line

            if best_surplus_node == -1:
                break  # Node is stranded

            # Trace path backwards
            curr = best_surplus_node
            path_len = 0

            while curr != n and curr != -1:
                path_nodes[path_len] = curr
                nxt = parent_node[curr]
                path_lines[path_len] = parent_line[curr]
                path_len += 1
                curr = nxt

            # Calculate max initial send to respect capacities and zero-crossings
            max_initial_send = Surplust[best_surplus_node]
            cum_eff = 1.0

            for i in range(path_len):
                sender = path_nodes[i]
                line = path_lines[i]
                is_fwd = (sender == solution.static.network[line, 0])

                avail_cap = cap_fwd[line] if is_fwd else cap_rev[line]
                edge_e = eff_fwd[line] if is_fwd else eff_rev[line]

                bottleneck_send = avail_cap / cum_eff
                if bottleneck_send < max_initial_send:
                    max_initial_send = bottleneck_send
                cum_eff *= edge_e

            # Scale if received power overfills the deficit
            received = max_initial_send * cum_eff
            if received > Fillt[n]:
                received = Fillt[n]
                max_initial_send = received / cum_eff

            # Apply physical transfers
            current_flow = max_initial_send
            Surplust[best_surplus_node] -= current_flow
            total_surplus -= current_flow

            for i in range(path_len):
                sender = path_nodes[i]
                line = path_lines[i]
                receiver = parent_node[sender]

                is_fwd = (sender == solution.static.network[line, 0])
                edge_e = eff_fwd[line] if is_fwd else eff_rev[line]

                # Update nodal boundary injections (Preserves UpdateUnbalancedt logic)
                Exportt[sender] -= current_flow
                Importt[receiver] += current_flow * edge_e

                # Update line netflow state
                if is_fwd:
                    if netflowt[line] < -TOLERANCE:  # Processing counter-flow
                        netflowt[line] += current_flow * edge_e
                    else:
                        netflowt[line] += current_flow
                else:
                    if netflowt[line] > TOLERANCE:  # Processing counter-flow
                        netflowt[line] -= current_flow * edge_e
                    else:
                        netflowt[line] -= current_flow

                # Update line capacities
                if netflowt[line] < -TOLERANCE:
                    cap_fwd[line] = -netflowt[line] * line_eff
                    eff_fwd[line] = 1.0 / line_eff
                else:
                    cap_fwd[line] = solution.assets.Clines[line] - netflowt[line]
                    eff_fwd[line] = line_eff

                if netflowt[line] > TOLERANCE:
                    cap_rev[line] = netflowt[line] * line_eff
                    eff_rev[line] = 1.0 / line_eff
                else:
                    cap_rev[line] = solution.assets.Clines[line] + netflowt[line]
                    eff_rev[line] = line_eff

                current_flow *= edge_e

            Fillt[n] -= received

    return Importt, Exportt
