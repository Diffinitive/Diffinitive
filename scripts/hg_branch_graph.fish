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
            add_node($1)
            add_node($2)
        }

        function add_node(node) {
            if (node == "" || seen_node[node]) {
                return
            }

            seen_node[node] = 1
            node_count += 1
            nodes[node_count] = node
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

        function reachable_component_without_edge(source, target, skip_source, skip_target,     current, child, key) {
            if (source == target) {
                return 1
            }

            if (visited_component[source]) {
                return 0
            }
            visited_component[source] = 1

            for (key in component_edge) {
                split(key, edge_parts, SUBSEP)
                current = edge_parts[1]
                child = edge_parts[2]

                if (current != source) {
                    continue
                }

                if (current == skip_source && child == skip_target) {
                    continue
                }

                if (reachable_component_without_edge(child, target, skip_source, skip_target)) {
                    return 1
                }
            }

            return 0
        }

        function edge_rank(source, source_component) {
            if (source == "default") {
                return 0
            }

            if (source == component_representative[source_component]) {
                return 1
            }

            return 2
        }

        function prefer_edge(source, target, previous_source, previous_target, source_component,     rank, previous_rank) {
            if (previous_source == "") {
                return 1
            }

            rank = edge_rank(source, source_component)
            previous_rank = edge_rank(previous_source, source_component)

            if (rank < previous_rank) {
                return 1
            }

            if (rank > previous_rank) {
                return 0
            }

            if (source < previous_source) {
                return 1
            }

            if (source > previous_source) {
                return 0
            }

            return target < previous_target
        }

        END {
            for (i = 1; i <= node_count; i++) {
                source = nodes[i]

                if (component[source] != "") {
                    continue
                }

                component_count += 1
                component[source] = component_count
                component_representative[component_count] = source

                for (j = i + 1; j <= node_count; j++) {
                    target = nodes[j]

                    if (component[target] != "") {
                        continue
                    }

                    delete visited
                    source_reaches_target = reachable_without_edge(source, target, "", "")
                    delete visited
                    target_reaches_source = reachable_without_edge(target, source, "", "")

                    if (source_reaches_target && target_reaches_source) {
                        component[target] = component_count
                    }
                }
            }

            for (i = 1; i <= node_count; i++) {
                source = nodes[i]
                source_component = component[source]

                if (component_representative[source_component] == "default") {
                    continue
                }

                if (source == "default" || source < component_representative[source_component]) {
                    component_representative[source_component] = source
                }
            }

            for (key in edge) {
                split(key, edge_parts, SUBSEP)
                source = edge_parts[1]
                target = edge_parts[2]
                source_component = component[source]
                target_component = component[target]

                if (source_component == target_component) {
                    reduced_edge[source SUBSEP target] = 1
                    continue
                }

                component_edge[source_component SUBSEP target_component] = 1

                if (prefer_edge(source, target, component_edge_source[source_component SUBSEP target_component], component_edge_target[source_component SUBSEP target_component], source_component)) {
                    component_edge_source[source_component SUBSEP target_component] = source
                    component_edge_target[source_component SUBSEP target_component] = target
                }
            }

            for (key in component_edge) {
                split(key, edge_parts, SUBSEP)
                source = edge_parts[1]
                target = edge_parts[2]

                delete visited_component
                if (!reachable_component_without_edge(source, target, source, target)) {
                    reduced_edge[component_edge_source[key] SUBSEP component_edge_target[key]] = 1
                }
            }

            for (key in reduced_edge) {
                split(key, edge_parts, SUBSEP)
                print edge_parts[1] "\t" edge_parts[2]
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
    ' "$rev_file" | sort -u
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

function compute_reduced_branch_dependencies --description 'Print reduced open-branch dependencies as source<TAB>target'
    compute_branch_dependencies | reduce_branch_dependencies
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

function __branch_graph_has_children --argument-names node edges_file
    set -l children (__branch_graph_children "$node" "$edges_file")

    test (count $children) -gt 0
end

function __branch_graph_single_child --argument-names node edges_file
    set -l children (__branch_graph_children "$node" "$edges_file")

    if test (count $children) -eq 1
        printf '%s\n' "$children[1]"
    end
end

function __branch_graph_parent_count --argument-names node edges_file
    set -l parents (__branch_graph_parents "$node" "$edges_file")

    count $parents
end

function __branch_graph_subtree_stops_at_path --argument-names node edges_file
    set -l path $argv[3..-1]
    set -l next_path $path $node
    set -l children (__branch_graph_children "$node" "$edges_file")

    if test (count $children) -eq 0
        return 0
    end

    for child in $children
        if contains -- "$child" $next_path
            continue
        end

        if not __branch_graph_subtree_stops_at_path "$child" "$edges_file" $next_path
            return 1
        end
    end

    return 0
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
    set -l path $argv[4..-1]
    set -l children (__branch_graph_children $node $edges_file)
    set -l candidates (__branch_graph_descendants "$node" "$edges_file" $path | sort -ur)

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

            __branch_graph_print_node "$child" "$edges_file" (math $depth + 1) $path $node

            if __branch_graph_subtree_stops_at_path "$child" "$edges_file" $path $node
                printf '%s|/\n' "$prefix"
            end
        end

        return 0
    end

    return 1
end

function __branch_graph_print_node --argument-names node edges_file depth
    set -l path $argv[4..-1]
    set -l prefix (__branch_graph_prefix $depth)
    set -l next_path $path $node

    if __branch_graph_try_print_two_parent_join "$node" "$edges_file" "$depth" $path
        return 0
    end

    if test $depth -gt 0
        set -l only_child (__branch_graph_single_child "$node" "$edges_file")

        if test -n "$only_child"
            if contains -- "$only_child" $next_path
                printf '%so %s (cycle)\n' "$prefix" "$only_child"
                printf '%s|\n' "$prefix"
                printf '%so %s\n' "$prefix" "$node"
                return 0
            end

            if test (__branch_graph_parent_count "$only_child" "$edges_file") -eq 1
                __branch_graph_print_node "$only_child" "$edges_file" "$depth" $next_path
                printf '%s|\n' "$prefix"
                printf '%so %s\n' "$prefix" "$node"
                return 0
            end
        end
    end

    printf '%so %s\n' "$prefix" "$node"

    set -l children (__branch_graph_children $node $edges_file)
    set -l first_child 1

    for child in $children
        if test $first_child -eq 1
            printf '%s|\n' "$prefix"
            set first_child 0
        end

        if contains -- $child $next_path
            if test $depth -gt 0
                printf '%s  o %s (cycle)\n' "$prefix" "$child"
                printf '%s /\n' "$prefix"
                continue
            else
                printf '%so %s (cycle)\n' (__branch_graph_prefix (math $depth + 1)) "$child"
            end
            printf '%s|/\n' "$prefix"
            continue
        end

        __branch_graph_print_node "$child" "$edges_file" (math $depth + 1) $next_path

        if __branch_graph_subtree_stops_at_path "$child" "$edges_file" $next_path
            printf '%s|/\n' "$prefix"
        end
    end
end

function __branch_graph_render_from_edges --argument-names edges_file
    set -l open_branches $argv[2..-1]
    set -l roots (__branch_graph_roots "$edges_file" $open_branches)
    set -l rendered_branches
    set -l printed_component 0

    if not __branch_graph_is_isolated default "$edges_file"
        __branch_graph_print_node default "$edges_file" 0 ''
        set rendered_branches $rendered_branches default (__branch_graph_descendants default "$edges_file")
        set printed_component 1
    end

    for root in $roots
        if contains -- "$root" $rendered_branches
            continue
        end
        if test (count $roots) -gt 1; and __branch_graph_is_isolated "$root" "$edges_file"
            continue
        end

        if test $printed_component -eq 1
            printf '\n'
        end
        __branch_graph_print_node "$root" "$edges_file" 0 ''
        set rendered_branches $rendered_branches "$root" (__branch_graph_descendants "$root" "$edges_file")
        set printed_component 1
    end

    for branch in $open_branches
        if contains -- "$branch" $rendered_branches
            continue
        end
        if __branch_graph_is_isolated "$branch" "$edges_file"
            continue
        end

        if test $printed_component -eq 1
            printf '\n'
        end
        __branch_graph_print_node "$branch" "$edges_file" 0 ''
        set rendered_branches $rendered_branches "$branch" (__branch_graph_descendants "$branch" "$edges_file")
        set printed_component 1
    end
end

function print_branch_dependency_graph --description 'Print the computed branch dependencies as an ASCII tree'
    set -l edges_file (mktemp)
    set -l open_branches (__hg_open_branches)

    begin
        compute_reduced_branch_dependencies
    end > "$edges_file"

    if test (count $open_branches) -eq 0
        rm -f $edges_file
        echo 'No open branches found.'
        return 1
    end

    __branch_graph_render_from_edges "$edges_file" $open_branches

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
    set -l open_branches (__hg_open_branches)

    if test (count $open_branches) -eq 0
        echo 'No open branches found.'
        return 1
    end

    compute_reduced_branch_dependencies
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

function __branch_graph_render_open_test --description 'Render a synthetic graph from edge lines and open branches'
    set -l separator_index (contains -i -- -- $argv)

    if test -z "$separator_index"
        echo 'Internal test error: missing -- separator.' >&2
        return 2
    end

    set -l edge_args
    if test $separator_index -gt 1
        set edge_args $argv[1..(math $separator_index - 1)]
    end

    set -l open_branches $argv[(math $separator_index + 1)..-1]
    set -l raw_edges_file (mktemp)
    set -l edges_file (mktemp)

    for edge in $edge_args
        printf '%s\n' "$edge"
    end > "$raw_edges_file"

    reduce_branch_dependencies < "$raw_edges_file" > "$edges_file"
    __branch_graph_render_from_edges "$edges_file" $open_branches

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

function __branch_graph_assert_open_render --argument-names name expected
    set -l actual_file (mktemp)
    set -l expected_file (mktemp)

    __branch_graph_render_open_test $argv[3..-1] > "$actual_file"
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

    __compute_branch_dependencies_from_files "$open_file" "$rev_file" | reduce_branch_dependencies > "$actual_file"
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

    set -l expected_reduced_cycle_dependencies 'default	feature
default	tooling
tooling	default'
    printf '%s\n' 'default	feature' 'default	tooling' 'tooling	default' 'tooling	feature' | __branch_graph_assert_dependencies 'keep outgoing dependency from cycle' "$expected_reduced_cycle_dependencies"
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
| o C
| |
| o D
|/
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

    set -l expected_rootless_default_cycle 'o default
|
| o default (cycle)
| |
| o tooling/mergemap
|/
| o examples
|/'
    __branch_graph_assert_open_render 'default cycle without graph root' "$expected_rootless_default_cycle" \
        'default	examples' \
        'default	tooling/mergemap' \
        'tooling/mergemap	default' \
        -- default examples tooling/mergemap
    or set failures (math $failures + 1)

    set -l expected_rootless_non_default_cycle 'o A
|
| o A (cycle)
| |
| o B
|/'
    __branch_graph_assert_open_render 'non-default cycle without graph root' "$expected_rootless_non_default_cycle" \
        'A	B' \
        'B	A' \
        -- A B
    or set failures (math $failures + 1)

    set -l expected_current_repository_reduced_dependencies 'default	examples
default	feature/grids/chart_normal
default	feature/grids/multiblock_grids
default	feature/lazy_tensors/pretty_printing
default	refactor/lazy_tensors/operator_simplifications
default	refactor/sbp_operators/boundary_operator_type_paramaters
default	refactor/sbpoperators/boundary_operators
default	tooling/mergemap
feature/grids/chart_normal	feature/sbp_operators/vector_operators
feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators
refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators
refactor/sbp_operators/boundary_operator_type_paramaters	refactor/lazy_tensors/adjoint
tooling/mergemap	default'
    printf '%s\n' \
        'default	examples' \
        'default	feature/grids/chart_normal' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/matrix_of_operators' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	feature/sbp_operators/vector_operators' \
        'default	refactor/lazy_tensors/adjoint' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbp_operators/boundary_operator_type_paramaters' \
        'default	refactor/sbpoperators/boundary_operators' \
        'default	tooling/mergemap' \
        'feature/grids/chart_normal	feature/sbp_operators/vector_operators' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/sbp_operators/vector_operators' \
        'refactor/sbp_operators/boundary_operator_type_paramaters	refactor/lazy_tensors/adjoint' \
        'tooling/mergemap	default' \
        'tooling/mergemap	refactor/lazy_tensors/adjoint' \
        'tooling/mergemap	refactor/sbp_operators/boundary_operator_type_paramaters' \
        | __branch_graph_assert_dependencies 'current repository reduced dependencies' "$expected_current_repository_reduced_dependencies"
    or set failures (math $failures + 1)

    set -l expected_current_repository_open_graph 'o default
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
| |
| o refactor/sbp_operators/boundary_operator_type_paramaters
|/
| o refactor/sbpoperators/boundary_operators
|/
| o default (cycle)
| |
| o tooling/mergemap
|/'
    __branch_graph_assert_open_render 'current repository graph with default cycle' "$expected_current_repository_open_graph" \
        'default	examples' \
        'default	feature/grids/chart_normal' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbp_operators/boundary_operator_type_paramaters' \
        'default	refactor/sbpoperators/boundary_operators' \
        'default	tooling/mergemap' \
        'feature/grids/chart_normal	feature/sbp_operators/vector_operators' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        'refactor/sbp_operators/boundary_operator_type_paramaters	refactor/lazy_tensors/adjoint' \
        'tooling/mergemap	default' \
        -- \
        tooling/mergemap \
        refactor/lazy_tensors/adjoint \
        feature/sbp_operators/vector_operators \
        refactor/sbpoperators/boundary_operators \
        feature/lazy_tensors/pretty_printing \
        feature/grids/multiblock_grids \
        examples \
        refactor/sbp_operators/boundary_operator_type_paramaters \
        default \
        feature/grids/chart_normal \
        feature/lazy_tensors/matrix_of_operators \
        refactor/lazy_tensors/operator_simplifications
    or set failures (math $failures + 1)

    set -l expected_current_repository_traction_reduced_dependencies 'default	examples
default	feature/grids/multiblock_grids
default	feature/lazy_tensors/pretty_printing
default	refactor/lazy_tensors/adjoint
default	refactor/sbpoperators/boundary_operators
feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators
feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons
refactor/lazy_tensors/adjoint	refactor/lazy_tensors/operator_simplifications
refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators'
    printf '%s\n' \
        'default	examples' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/matrix_of_operators' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	feature/sbp_operators/traction_conditons' \
        'default	feature/sbp_operators/vector_operators' \
        'default	refactor/lazy_tensors/adjoint' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbpoperators/boundary_operators' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/traction_conditons' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/adjoint	refactor/lazy_tensors/operator_simplifications' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/operator_simplifications	feature/sbp_operators/vector_operators' \
        | __branch_graph_assert_dependencies 'current repository traction reduced dependencies' "$expected_current_repository_traction_reduced_dependencies"
    or set failures (math $failures + 1)

    set -l expected_current_repository_traction_graph 'o default
|
| o refactor/sbpoperators/boundary_operators
|/
| o feature/sbp_operators/traction_conditons
| |
| o feature/sbp_operators/vector_operators
| |
| o feature/lazy_tensors/matrix_of_operators
| |
| o refactor/lazy_tensors/operator_simplifications
| |
| o refactor/lazy_tensors/adjoint
|/
| o feature/lazy_tensors/pretty_printing
|/
| o feature/grids/multiblock_grids
|/
| o examples
|/'
    __branch_graph_assert_open_render 'current repository graph with traction chain' "$expected_current_repository_traction_graph" \
        'default	examples' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	refactor/lazy_tensors/adjoint' \
        'default	refactor/sbpoperators/boundary_operators' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/adjoint	refactor/lazy_tensors/operator_simplifications' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        -- \
        default \
        refactor/lazy_tensors/adjoint \
        feature/sbp_operators/traction_conditons \
        refactor/lazy_tensors/operator_simplifications \
        refactor/sbpoperators/boundary_operators \
        feature/lazy_tensors/pretty_printing \
        feature/grids/multiblock_grids \
        examples \
        feature/sbp_operators/vector_operators \
        feature/lazy_tensors/matrix_of_operators
    or set failures (math $failures + 1)

    set -l expected_current_repository_operator_fusing_reduced_dependencies 'default	bugfix/sbp_operators/second_derivative_variable/equality
default	examples
default	feature/grids/multiblock_grids
default	feature/lazy_tensors/pretty_printing
default	refactor/lazy_tensors/operator_simplifications
default	refactor/sbpoperators/boundary_operators
default	tooling/mergemap
feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators
feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons
refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators
refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/operator_fusing
tooling/mergemap	default'
    printf '%s\n' \
        'default	bugfix/sbp_operators/second_derivative_variable/equality' \
        'default	examples' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/matrix_of_operators' \
        'default	feature/lazy_tensors/operator_fusing' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	feature/sbp_operators/traction_conditons' \
        'default	feature/sbp_operators/vector_operators' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbpoperators/boundary_operators' \
        'default	tooling/mergemap' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/traction_conditons' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/operator_fusing' \
        'refactor/lazy_tensors/operator_simplifications	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/operator_simplifications	feature/sbp_operators/vector_operators' \
        'tooling/mergemap	bugfix/sbp_operators/second_derivative_variable/equality' \
        'tooling/mergemap	default' \
        'tooling/mergemap	feature/lazy_tensors/operator_fusing' \
        'tooling/mergemap	feature/sbp_operators/traction_conditons' \
        'tooling/mergemap	feature/sbp_operators/vector_operators' \
        'tooling/mergemap	refactor/lazy_tensors/operator_simplifications' \
        | __branch_graph_assert_dependencies 'current repository operator fusing reduced dependencies' "$expected_current_repository_operator_fusing_reduced_dependencies"
    or set failures (math $failures + 1)

    set -l expected_current_repository_operator_fusing_graph 'o default
|
| o default (cycle)
| |
| o tooling/mergemap
|/
| o refactor/sbpoperators/boundary_operators
|/
| o refactor/lazy_tensors/operator_simplifications
| |
| | o feature/lazy_tensors/operator_fusing
| |/
| | o feature/sbp_operators/traction_conditons
| | |
| | o feature/sbp_operators/vector_operators
| | |
| | o feature/lazy_tensors/matrix_of_operators
| |/
|/
| o feature/lazy_tensors/pretty_printing
|/
| o feature/grids/multiblock_grids
|/
| o examples
|/
| o bugfix/sbp_operators/second_derivative_variable/equality
|/'
    __branch_graph_assert_open_render 'current repository graph with operator fusing branch' "$expected_current_repository_operator_fusing_graph" \
        'default	bugfix/sbp_operators/second_derivative_variable/equality' \
        'default	examples' \
        'default	feature/grids/multiblock_grids' \
        'default	feature/lazy_tensors/pretty_printing' \
        'default	refactor/lazy_tensors/operator_simplifications' \
        'default	refactor/sbpoperators/boundary_operators' \
        'default	tooling/mergemap' \
        'feature/lazy_tensors/matrix_of_operators	feature/sbp_operators/vector_operators' \
        'feature/sbp_operators/vector_operators	feature/sbp_operators/traction_conditons' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/matrix_of_operators' \
        'refactor/lazy_tensors/operator_simplifications	feature/lazy_tensors/operator_fusing' \
        'tooling/mergemap	default' \
        -- \
        default \
        bugfix/sbp_operators/second_derivative_variable/equality \
        examples \
        feature/grids/multiblock_grids \
        feature/lazy_tensors/operator_fusing \
        feature/lazy_tensors/pretty_printing \
        feature/sbp_operators/traction_conditons \
        feature/sbp_operators/vector_operators \
        refactor/lazy_tensors/operator_simplifications \
        refactor/sbpoperators/boundary_operators \
        tooling/mergemap \
        feature/lazy_tensors/matrix_of_operators
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
