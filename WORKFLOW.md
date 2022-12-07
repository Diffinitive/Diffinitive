# Branch model

The `default` branch contains stable code and the goal is that all tests should be passing on this branch. Version of the code are marked with mercurial tags. Changes in the code are developed in named branches which are closed and merged into `default` when the code is ready. During development merging `default` into the development branch is encouraged to avoid complicated conflicts.

Branches are named using slash separated keywords. The first keyword describes the type of change being pursued in the branch. Important type keywords are
 * feature
 * bugfix
 * refactor
Further keywords may describe where, e.g. what sub package, the change happens. The last keyword should describe the change.

Some examples:
 * refactor/grids: Branch to refactor the grids module
 * bugfix/lazy_tensors/lazyfunctionarray: Branch to fix a bug in LazyFunctionArray

## Merging a branch into `default`
The changes in a branch has been reviewed and deemed ready to merge the branch is closed and then merged.

Before merging a development branch, `default` should be merge into the development branch to make sure the whole state of the code is reviewed and tested before it ends up on `default`.

With the development branch active the following commands can be used to complete the merging of a development branch.
```shell
hg merge default
hg commit --close-branch -m "Close before merge"
hg update default
hg merge development/branch
hg commit -m "Merge development/branch"
```

# Review

## Checklist for review
* Push and pull new changes
* Search and check TODOs
* Search and check TBDs
* Search and check REVIEWs
* Review code
* Review tests
* Review docstrings
* Render Documenter and check docstrings in browser
* Run full tests

# Special comments
The following special comments are used:
* `# TODO: `: Something that should be done at some point.
* `# TBD: `:  "To be determined", i.e a decision that has to be made.
* `# REVIEW: `: A review comment. Should only exist on development branches.
