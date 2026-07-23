#!/usr/bin/env fish

# Print an ASCII dependency graph for open Mercurial branches in the current
# repository. Edges point from the nearest open ancestor branch to the dependent
# open branch, with closed intermediate branches treated as transparent ancestry.
#
# Usage:
#   scripts/hg_branch_graph.fish
#       Render the open-branch dependency graph.
#   scripts/hg_branch_graph.fish --dependencies
#       Print the branch dependencies as source<TAB>target.
#   scripts/hg_branch_graph.fish --reduced-dependencies
#       Print the reduced branch dependencies as source<TAB>target.
#   scripts/hg_branch_graph.fish --test
#       Run the script's synthetic rendering and dependency tests.
#   scripts/hg_branch_graph.fish --help
#       Show the command-line help.
#
# The graph is a summary of branch relationships, not a replacement for
# `hg log -G`: it collapses revisions within each branch and removes dependency
# edges already implied by an indirect path.

function __branch_graph_help --description 'Print command usage'
    printf '%s\n' \
        'Print an ASCII dependency graph for open Mercurial branches.' \
        '' \
        'Usage:' \
        '  hg_branch_graph.fish [--help|-h]' \
        '  hg_branch_graph.fish [--dependencies|--deps]' \
        '  hg_branch_graph.fish [--reduced-dependencies|--reduced-deps]' \
        '  hg_branch_graph.fish [--test|test]' \
        '' \
        'With no arguments, renders the branch dependency graph for the current repository.' \
        'Use --dependencies to print branch dependencies as source<TAB>target.' \
        'Use --reduced-dependencies to print reduced branch dependencies as source<TAB>target.' \
        'Use --test to run the built-in graph rendering and dependency tests.'
end

function __hg_open_branches --description 'Print open Mercurial branches, one per line'
    hg branches --template '{branch}\n'
end

function reduce_branch_dependencies --description 'Remove dependency edges covered by an indirect dependency path'
    awk -F '\t' '
        NF >= 2 {
            edge[$1 SUBSEP $2] = 1
        }

        function reachable_without_edge(source, target, skip_source, skip_target,     current, child, key) {
            if (source == target) {
                return 1
            }

            if (visited[source]) {
                return 0
            }
            visited[source] = 1

            for (key in edge) {
                split(key, edge_parts, SUBSEP)
                current = edge_parts[1]
                child = edge_parts[2]

                if (current != source) {
                    continue
                }

                if (current == skip_source && child == skip_target) {
                    continue
                }

                if (reachable_without_edge(child, target, skip_source, skip_target)) {
                    return 1
                }
            }

            return 0
        }

        END {
            for (key in edge) {
                split(key, edge_parts, SUBSEP)
                source = edge_parts[1]
                target = edge_parts[2]

                delete visited
                if (!reachable_without_edge(source, target, source, target)) {
                    print source "\t" target
                }
            }
        }
    ' | sort -u
end

function __compute_branch_dependencies_from_files --argument-names open_file rev_file
    awk -F '\t' -v open_file="$open_file" '
        BEGIN {
            while ((getline line < open_file) > 0) {
                open[line] = 1
            }
            close(open_file)
        }

        function add_label(labels, label,     separator) {
            if (label == "") {
                return labels
            }

            if (seen_label[label]) {
                return labels
            }

            seen_label[label] = 1
            separator = labels == "" ? "" : SUBSEP
            return labels separator label
        }

        function add_labels(labels, source,     count, i, labels_parts) {
            count = split(source, labels_parts, SUBSEP)
            for (i = 1; i <= count; i++) {
                labels = add_label(labels, labels_parts[i])
            }

            return labels
        }

        function print_dependencies(labels, branch,     count, i, labels_parts) {
            count = split(labels, labels_parts, SUBSEP)
            for (i = 1; i <= count; i++) {
                if (labels_parts[i] != "" && labels_parts[i] != branch && open[labels_parts[i]]) {
                    print labels_parts[i] "\t" branch
                }
            }
        }

        {
            rev = $1
            branch = $2
            p1_rev = $3
            p2_rev = $4

            if (!open[branch]) {
                delete seen_label
                labels = ""
                labels = add_labels(labels, rev_labels[p1_rev])
                labels = add_labels(labels, rev_labels[p2_rev])
                rev_labels[rev] = labels
                next
            }

            print_dependencies(rev_labels[p1_rev], branch)
            print_dependencies(rev_labels[p2_rev], branch)

            delete seen_label
            labels = ""
            labels = add_labels(labels, rev_labels[p1_rev])
            labels = add_labels(labels, rev_labels[p2_rev])
            labels = add_label(labels, branch)
            rev_labels[rev] = labels
        }
    ' "$rev_file" | sort -u | reduce_branch_dependencies
end

function compute_branch_dependencies --description 'Print open-branch dependencies as source<TAB>target'
    set -l open_branches (__hg_open_branches)

    if test (count $open_branches) -eq 0
        return 0
    end

    set -l open_file (mktemp)
    set -l rev_file (mktemp)
    printf '%s\n' $open_branches > $open_file
    hg log -r 'sort(all(), rev)' --template '{rev}\t{branch}\t{p1rev}\t{p2rev}\n' > $rev_file

    __compute_branch_dependencies_from_files "$open_file" "$rev_file"

    rm -f $open_file $rev_file
end

function __branch_graph_children --argument-names node edges_file
    while read -l line
        test -n "$line"; or continue

        set -l fields (string split \t -- $line)
        if test "$fields[1]" = "$node"
            echo $fields[2]
        end
    end < "$edges_file" | sort -ur
end

function __branch_graph_parents --argument-names node edges_file
    while read -l line
        test -n "$line"; or continue

        set -l fields (string split \t -- $line)
        if test "$fields[2]" = "$node"
            echo $fields[1]
        end
    end < "$edges_file" | sort -ur
end

function __branch_graph_path_to --argument-names source target edges_file
    set -l path $argv[4..-1]
    set -l next_path $path $source

    if test "$source" = "$target"
        printf '%s\n' $next_path
        return 0
    end

    for child in (__branch_graph_children "$source" "$edges_file")
        if contains -- "$child" $next_path
            continue
        end

        set -l child_path (__branch_graph_path_to "$child" "$target" "$edges_file" $next_path)
        if test (count $child_path) -gt 0
            printf '%s\n' $child_path
            return 0
        end
    end

    return 1
end

function __branch_graph_descendants --argument-names node edges_file
    set -l path $argv[3..-1]
    set -l next_path $path $node

    for child in (__branch_graph_children "$node" "$edges_file")
        if contains -- "$child" $next_path
            continue
        end

        echo "$child"
        __branch_graph_descendants "$child" "$edges_file" $next_path
    end
end

function __branch_graph_roots --argument-names edges_file
    set -l targets

    while read -l line
        test -n "$line"; or continue

        set -l fields (string split \t -- $line)
        set targets $targets $fields[2]
    end < "$edges_file"

    set -l open_branches $argv[2..-1]

    for node in $open_branches
        if contains -- $node $targets
            continue
        end
        echo $node
    end
end

function __branch_graph_is_isolated --argument-names node edges_file
    set -l children (__branch_graph_children "$node" "$edges_file")
    set -l parents (__branch_graph_parents "$node" "$edges_file")

    test (count $children) -eq 0; and test (count $parents) -eq 0
end

function __branch_graph_prefix --argument-names depth
    set -l prefix ''

    if test $depth -le 0
        return 0
    end

    for i in (seq 1 $depth)
        set prefix "$prefix| "
    end

    printf '%s' "$prefix"
end

function __branch_graph_try_print_two_parent_join --argument-names node edges_file depth
    set -l children (__branch_graph_children $node $edges_file)
    set -l candidates (__branch_graph_descendants "$node" "$edges_file" | sort -ur)

    for candidate in $candidates
        set -l parents (__branch_graph_parents $candidate $edges_file)
        set -l parent_paths
        set -l branch_roots

        for parent in $parents
            set -l path_to_parent (__branch_graph_path_to "$node" "$parent" "$edges_file")

            if test (count $path_to_parent) -gt 1
                set parent_paths $parent_paths (string join \t -- $path_to_parent)
                set branch_roots $branch_roots $path_to_parent[2]
            end
        end

        if test (count $parent_paths) -ne 2
            continue
        end

        set -l prefix (__branch_graph_prefix $depth)
        set -l child_prefix (__branch_graph_prefix (math $depth + 1))
        set -l upper_path (string split \t -- $parent_paths[1])
        set -l lower_path (string split \t -- $parent_paths[2])
        set -l upper_chain $upper_path[2..-1]
        set -l lower_chain $lower_path[2..-1]

        printf '%so %s\n' "$prefix" "$node"
        printf '%s|\n' "$prefix"
        printf '%s  o %s\n' "$child_prefix" "$candidate"
        printf '%s /|\n' "$child_prefix"

        for index in (seq (count $upper_chain) -1 1)
            if test $index -ne (count $upper_chain)
                printf '%s| |\n' "$child_prefix"
            end
            printf '%so | %s\n' "$child_prefix" "$upper_chain[$index]"
        end

        printf '%s|/  |\n' "$prefix"

        for index in (seq (count $lower_chain) -1 1)
            if test $index -ne (count $lower_chain)
                printf '%s  |\n' "$child_prefix"
            end
            printf '%s  o %s\n' "$child_prefix" "$lower_chain[$index]"
        end

        printf '%s /\n' "$child_prefix"
        printf '%s| /\n' "$prefix"
        printf '%s|/\n' "$prefix"

        set -l used_roots (printf '%s\n' $branch_roots | sort -u)
        set -l remaining_children
        for child in $children
            if contains -- "$child" $used_roots
                continue
            end
            set remaining_children $remaining_children $child
        end

        for child in (printf '%s\n' $remaining_children | sort)
            test -n "$child"; or continue

            __branch_graph_print_node "$child" "$edges_file" (math $depth + 1) $node

            set -l grandchildren (__branch_graph_children "$child" "$edges_file")
            if test (count $grandchildren) -eq 0
                printf '%s|/\n' "$prefix"
            end
        end

        return 0
    end

    return 1
end

function __branch_graph_print_node --argument-names node edges_file depth
    set -l path $argv[4..-1]

    if __branch_graph_try_print_two_parent_join "$node" "$edges_file" "$depth"
        return 0
    end

    set -l prefix (__branch_graph_prefix $depth)
    printf '%so %s\n' "$prefix" "$node"

    set -l next_path $path $node
    set -l children (__branch_graph_children $node $edges_file)
    set -l first_child 1

    for child in $children
        if test $first_child -eq 1
            printf '%s|\n' "$prefix"
            set first_child 0
        end

        if contains -- $child $next_path
            printf '%so %s (cycle)\n' (__branch_graph_prefix (math $depth + 1)) "$child"
            printf '%s|/\n' "$prefix"
            continue
        end

        __branch_graph_print_node "$child" "$edges_file" (math $depth + 1) $next_path

        set -l grandchildren (__branch_graph_children $child $edges_file)
        if test (count $grandchildren) -eq 0
            printf '%s|/\n' "$prefix"
        end
    end
end

function print_branch_dependency_graph --description 'Print the computed branch dependencies as an ASCII tree'
    set -l edges_file (mktemp)
    set -l open_branches (__hg_open_branches)

    begin
        compute_branch_dependencies
    end > "$edges_file"

    if test (count $open_branches) -eq 0
        rm -f $edges_file
        echo 'No open branches found.'
        return 1
    end

    set -l roots (__branch_graph_roots "$edges_file" $open_branches)

    if contains -- default $roots
        if test (count $roots) -eq 1; or not __branch_graph_is_isolated default "$edges_file"
            __branch_graph_print_node default "$edges_file" 0 ''
        end
    end

    for root in $roots
        if test "$root" = default
            continue
        end
        if test (count $roots) -gt 1; and __branch_graph_is_isolated "$root" "$edges_file"
            continue
        end
        __branch_graph_print_node "$root" "$edges_file" 0 ''
    end

    rm -f "$edges_file"
end

function print_branch_dependency_information --description 'Print the computed branch dependencies as source<TAB>target'
    set -l open_branches (__hg_open_branches)

    if test (count $open_branches) -eq 0
        echo 'No open branches found.'
        return 1
    end

    compute_branch_dependencies
end

function print_reduced_branch_dependency_information --description 'Print the computed reduced branch dependencies as source<TAB>target'
    print_branch_dependency_information
end

function __branch_graph_render_test --description 'Render a synthetic graph from edge lines and roots'
    set -l separator_index (contains -i -- -- $argv)

    if test -z "$separator_index"
        echo 'Internal test error: missing -- separator.' >&2
        return 2
    end

    set -l edge_args
    if test $separator_index -gt 1
        set edge_args $argv[1..(math $separator_index - 1)]
    end

    set -l roots $argv[(math $separator_index + 1)..-1]
    set -l raw_edges_file (mktemp)
    set -l edges_file (mktemp)

    for edge in $edge_args
        printf '%s\n' "$edge"
    end > "$raw_edges_file"

    reduce_branch_dependencies < "$raw_edges_file" > "$edges_file"

    for root in $roots
        if test (count $roots) -gt 1; and __branch_graph_is_isolated "$root" "$edges_file"
            continue
        end
        __branch_graph_print_node "$root" "$edges_file" 0 ''
    end

    rm -f "$raw_edges_file" "$edges_file"
end

function __branch_graph_assert_render --argument-names name expected
    set -l actual_file (mktemp)
    set -l expected_file (mktemp)

    __branch_graph_render_test $argv[3..-1] > "$actual_file"
    printf '%s\n' "$expected" > "$expected_file"

    if diff -u "$expected_file" "$actual_file" >/dev/null
        rm -f "$actual_file" "$expected_file"
        printf 'ok - %s\n' "$name"
        return 0
    end

    printf 'not ok - %s\n' "$name" >&2
    diff -u "$expected_file" "$actual_file" >&2
    rm -f "$actual_file" "$expected_file"
    return 1
end

function __branch_graph_assert_dependencies --argument-names name expected
    set -l actual_file (mktemp)
    set -l expected_file (mktemp)

    reduce_branch_dependencies > "$actual_file"
    printf '%s\n' "$expected" > "$expected_file"

    if diff -u "$expected_file" "$actual_file" >/dev/null
        rm -f "$actual_file" "$expected_file"
        printf 'ok - %s\n' "$name"
        return 0
    end

    printf 'not ok - %s\n' "$name" >&2
    diff -u "$expected_file" "$actual_file" >&2
    rm -f "$actual_file" "$expected_file"
    return 1
end

function __branch_graph_assert_computed_dependencies --argument-names name expected
    set -l separator_index (contains -i -- -- $argv)

    if test -z "$separator_index"
        echo 'Internal test error: missing -- separator.' >&2
        return 2
    end

    set -l open_branches
    if test $separator_index -gt 3
        set open_branches $argv[3..(math $separator_index - 1)]
    end

    set -l open_file (mktemp)
    set -l rev_file (mktemp)
    set -l actual_file (mktemp)
    set -l expected_file (mktemp)

    printf '%s\n' $open_branches > "$open_file"
    printf '%s\n' $argv[(math $separator_index + 1)..-1] > "$rev_file"

    __compute_branch_dependencies_from_files "$open_file" "$rev_file" > "$actual_file"
    printf '%s\n' "$expected" > "$expected_file"

    if diff -u "$expected_file" "$actual_file" >/dev/null
        rm -f "$open_file" "$rev_file" "$actual_file" "$expected_file"
        printf 'ok - %s\n' "$name"
        return 0
    end

    printf 'not ok - %s\n' "$name" >&2
    diff -u "$expected_file" "$actual_file" >&2
    rm -f "$open_file" "$rev_file" "$actual_file" "$expected_file"
    return 1
end

function test_branch_dependency_graph --description 'Run synthetic graph rendering tests'
    set -l failures 0

    set -l expected_reduced_dependencies 'A	B
B	C'
    printf '%s\n' 'A	B' 'A	C' 'B	C' | __branch_graph_assert_dependencies 'remove transitive dependency edge' "$expected_reduced_dependencies"
    or set failures (math $failures + 1)

    set -l expected_single 'o default'
    __branch_graph_assert_render 'single node' "$expected_single" -- default
    or set failures (math $failures + 1)

    set -l expected_one_child 'o default
|
| o A
|/'
    __branch_graph_assert_render 'one child' "$expected_one_child" 'default	A' -- default
    or set failures (math $failures + 1)

    set -l expected_siblings 'o default
|
| o B
|/
| o A
|/'
    __branch_graph_assert_render 'sibling children' "$expected_siblings" 'default	A' 'default	B' -- default
    or set failures (math $failures + 1)

    set -l expected_nested 'o default
|
| o D
| |
| | o C
| |/
| o B
|/
| o A
|/'
    __branch_graph_assert_render 'nested branch' "$expected_nested" 'default	A' 'default	B' 'default	D' 'D	C' -- default
    or set failures (math $failures + 1)

    set -l expected_diamond 'o default
|
|   o C
|  /|
| o | B
|/  |
|   o A
|  /
| /
|/'
    __branch_graph_assert_render 'A and B merge into C' "$expected_diamond" 'default	A' 'default	B' 'A	C' 'B	C' -- default
    or set failures (math $failures + 1)

    set -l expected_three_parent_merge 'o default
|
|   o C
|  /|
| o | B
|/  |
|   o A
|  /
| /
|/'
    __branch_graph_assert_render 'A, B, and default merge into C' "$expected_three_parent_merge" 'default	A' 'default	B' 'default	C' 'A	C' 'B	C' -- default
    or set failures (math $failures + 1)

    set -l expected_current_repository_dependencies 'default	examples
default	feature/grids/chart_normal
default	feature/grids/multiblock_grids
default	feature/lazy_tensors/pretty_printing
default	refactor/lazy_tensors/adjoint
default	refactor/lazy_tensors/operator_simplifications
default	refactor/sbpoperators/boundary_operators
default	tooling/mergemap
feature/grids/chart_normal	feature/sbp_operators/vector_operators
feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators
refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators'
    __branch_graph_assert_computed_dependencies 'current repository dependencies through closed branches' "$expected_current_repository_dependencies" \
        default \
        examples \
        feature/grids/chart_normal \
        feature/grids/multiblock_grids \
        feature/lazy_tensors/matrix_of_operators \
        feature/lazy_tensors/pretty_printing \
        feature/sbp_operators/vector_operators \
        refactor/lazy_tensors/adjoint \
        refactor/lazy_tensors/operator_simplifications \
        refactor/sbpoperators/boundary_operators \
        tooling/mergemap \
        -- \
        '1	default	-1	-1' \
        '2	refactor/lazy_tensors/operator_simplifications	1	-1' \
        '3	feature/grids/with_jacobian	1	-1' \
        '4	feature/grids/chart_normal	3	-1' \
        '5	feature/lazy_tensors/matrix_of_operators	1	2' \
        '6	feature/sbp_operators/vector_operators	1	-1' \
        '7	feature/sbp_operators/vector_operators	6	4' \
        '8	feature/sbp_operators/vector_operators	7	5' \
        '9	feature/grids/multiblock_grids	1	-1' \
        '10	feature/lazy_tensors/pretty_printing	1	-1' \
        '11	refactor/lazy_tensors/adjoint	1	-1' \
        '12	refactor/sbpoperators/boundary_operators	1	-1' \
        '13	examples	1	-1' \
        '14	default	1	-1' \
        '15	tooling/mergemap	14	-1'
    or set failures (math $failures + 1)

    set -l expected_current_repository_after_merge 'o default
|
|   o feature/sbp_operators/vector_operators
|  /|
| o | feature/lazy_tensors/matrix_of_operators
| | |
| o | refactor/lazy_tensors/operator_simplifications
|/  |
|   o feature/grids/chart_normal
|  /
| /
|/
| o examples
|/
| o feature/grids/multiblock_grids
|/
| o feature/lazy_tensors/pretty_printing
|/
| o refactor/lazy_tensors/adjoint
|/
| o refactor/sbpoperators/boundary_operators
|/
| o tooling/mergemap
|/'
    __branch_graph_assert_render 'current repository graph after merge' "$expected_current_repository_after_merge" \
        'default	examples' \
        'default	feature/grids/chart_normal' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	refactor/lazy_tensors/adjoint' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbpoperators/boundary_operators' \
        'default	tooling/mergemap' \
        'feature/grids/chart_normal	feature/sbp_operators/vector_operators' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        -- default
    or set failures (math $failures + 1)

    set -l expected_current_repository 'o default
|
|   o feature/sbp_operators/vector_operators
|  /|
| o | feature/lazy_tensors/matrix_of_operators
| | |
| o | refactor/lazy_tensors/operator_simplifications
|/  |
|   o feature/grids/chart_normal
|   |
|   o feature/grids/with_jacobian
|  /
| /
|/
| o feature/grids/multiblock_grids
|/
| o feature/lazy_tensors/pretty_printing
|/
| o refactor/lazy_tensors/adjoint
|/
| o refactor/sbpoperators/boundary_operators
|/'
    __branch_graph_assert_render 'current repository graph' "$expected_current_repository" \
        'default	feature/grids/multiblock_grids' \
        'default	feature/grids/with_jacobian' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	refactor/lazy_tensors/adjoint' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbpoperators/boundary_operators' \
        'feature/grids/chart_normal	feature/sbp_operators/vector_operators' \
        'feature/grids/with_jacobian	feature/grids/chart_normal' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        -- default examples
    or set failures (math $failures + 1)

    if test $failures -gt 0
        printf '%s test(s) failed.\n' "$failures" >&2
        return 1
    end

    printf 'All graph rendering tests passed.\n'
end

if not set -q HG_BRANCH_GRAPH_LIBRARY_ONLY
    switch "$argv[1]"
        case --help -h
            __branch_graph_help
        case test --test
            test_branch_dependency_graph
        case --dependencies --deps
            print_branch_dependency_information
        case --reduced-dependencies --reduced-deps
            print_reduced_branch_dependency_information
        case ''
            print_branch_dependency_graph
        case '*'
            printf 'Usage: %s [--help|-h] [--dependencies|--deps] [--reduced-dependencies|--reduced-deps] [--test|test]\n' (status filename) >&2
            exit 2
    end
end
