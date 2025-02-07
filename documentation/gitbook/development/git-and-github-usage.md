# Git and GitHub Usage

We now use a modified gitflow style workflow for working with git and GitHub. For a detailed overview of this topic please refer to [Atlassian's article on Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).


## New Features

New features should be developed in a new branch checked out from the `dev` branch and then merged back into the `dev` branch via a PR on GitHub when ready for review. These feature branches can be deleted after merging into `dev`, unless someone from operations requests that it be kept around. For example, operations may want to merge the feature into their operational branch to get new functionality in advance of a release. By convention feature branches should be prefixed with `feature/<GitHub issue>/`, I.e. `feature/99/cool-new-thing`. Feature branch should also include edits to the GitBook documentation that describe their changes.


## Hot Fixes

Hot fixes should be developed in a new branch checked out from the `main` branch and merged back into the `main` branch via a PR on GitHub when ready for review. After successfully merging into `main` the hot fix branch should then be merged into `dev`, making appropriate adjustments to stabilize the feature. The priority for hot fixes is to correct a major issue quickly, so it is okay to delay detailed testing/documentation until merging into `dev`. By convention hot fix branches should be prefixed with `hotfix/`, I.e. `hotfix/important-fix-to-something`, and then converted into a feature branch after merging into main. These do not have to include edits to the GitBook documentation, but if the hotfix conflicts with what is described in the GitBook documentation it's **strongly recommended**.


## Creating Releases

Periodically releases will be created by merging the `dev` branch into `main` via a PR on GitHub and creating a new release the `main` branch after merging. These PRs should avoid discussion of individual feature changes, those discussions should be reserved for and handled in the feature PRs. If there is a feature that poses a significant problem in the process of creating a new release those changes should be treated like a new feature. The main purpose of this PR is to:

1. Resolve merge conflicts generated by hot fixes,
2. Making minor edits to documentation to make it clearer or more cohesive, and
3. Updating the `NEWS.md` file with a summary of the changes included in the release.


## Operations

Operational work should be developed in a new branch checked out from the `main` branch if there are modifications needed to the `flepiMoP` codebase. Pre-released features can be merged directly into this operational branch from the corresponding feature branch as needed via a git merge or rebase not a GitHub PR. After the operational cycle is over, the operations branch **should not** be deleted, instead should be kept around for archival reasons. Operational work needs to move quickly and usually does not involve documenting or testing code and is therefore unsuitable for merging into `dev` or `main` directly. Instead potential features should be extracted from an operations branch into a feature branch using [git cherry-pick](https://git-scm.com/docs/git-cherry-pick) and then modified into an appropriates state for merging into `dev` like a feature branch. By convention operations branch names should be prefixed with `operations/`, I.e. `operations/flu-SMH-2023-24`.
