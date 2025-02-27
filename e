[33mcommit 9e9e555c7da075c323f3d1945dc35e66f3bdf2c6[m[33m ([m[1;36mHEAD -> [m[1;32mfeature/franka_pick_place[m[33m, [m[1;31morigin/feature/franka_pick_place[m[33m)[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Thu Feb 27 10:14:45 2025 +0100

    dropping reward iterations. still unsucessful.

[33mcommit e2de4cef7aa4752bbd803c4019c16d1ec4de5e79[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Wed Feb 26 14:56:10 2025 +0100

    adds reward scheduling and refining dropping reward. No successful drop yet

[33mcommit 51c486bc2fb71fb696eff03370ec1e28e650d9eb[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Fri Feb 21 17:50:59 2025 +0100

    adds dropping reward. Doesn't work properly

[33mcommit 2a2b515c679f098da337a959f9e3636797033f63[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Fri Feb 21 15:01:11 2025 +0100

    Good performance without table spawned.

[33mcommit 1818deb249378ad5765fabf9de41d9dffb8128f5[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Wed Feb 19 17:21:24 2025 +0100

    cleans up directory

[33mcommit 627197d13bf2e67648fb2eb582d3d1227db600e5[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Fri Feb 14 16:33:49 2025 +0100

    adds manager based pick plac environment

[33mcommit 9437aaeaaccda4854879cf2f5c6c21c6f56b51c4[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Thu Feb 13 16:55:35 2025 +0100

    adds target alignment reward

[33mcommit 7ebbafc67a877190810101fe8e0181210ccadd04[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Thu Feb 13 15:51:01 2025 +0100

    first successful lift to target

[33mcommit fab53d8f468b029281a2cff4c7c8167c52516c97[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Thu Feb 13 14:45:27 2025 +0100

    adds target reward

[33mcommit 1f4fd978106c1db4de38edbcf3b133a02466b2cd[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Thu Feb 13 10:10:49 2025 +0100

    adds target marker

[33mcommit 2f5246a34c8dccd715fe509f4da6d9ca540c4115[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Tue Feb 11 15:26:23 2025 +0100

    changes body_state_w to body_link_state_w

[33mcommit 34b843058f71378900149d32e7e67a1dcbee4da4[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Tue Feb 11 15:02:38 2025 +0100

    renames imports and variables to adhere to the latest version

[33mcommit a10b07f99d4d5365866d23f3ab449d6704055e3f[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Tue Feb 11 09:49:17 2025 +0100

    changes init to use new convention

[33mcommit 9c492d343da20707e7ce37322f21ce84340aeaaa[m
Author: NeilStrongarms <chris.haeberli@gmail.com>
Date:   Mon Feb 10 14:57:01 2025 +0100

    adds franka_pick_place files

[33mcommit c4bec8fe01c2fd83a0a25da184494b37b3e3eb61[m
Author: Mayank Mittal <12863862+Mayankm96@users.noreply.github.com>
Date:   Sun Feb 9 01:40:02 2025 +0100

    Switches to RSL-RL install from PyPI (#1811)
    
    # Description
    
    Since we now publish PyPI package for rsl-rl with its 2.1.1 release,
    this MR modifies the installation to use the PyPI package instead of the
    GitHub repository.
    
    Fixes [# (issue)](https://github.com/leggedrobotics/rsl_rl/issues/57)
    
    ## Type of change
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    ---------
    
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>

[33mcommit bf28d97486156f337a763c85790a44f85d374131[m
Author: samibouziri <79418773+samibouziri@users.noreply.github.com>
Date:   Fri Feb 7 19:24:39 2025 +0100

    Fixes no matching distribution found for rsl-rl (unavailable) (#1808)
    
    # Description
    Fixing the `No matching distribution found for rsl-rl (unavailable)`
    Error
    Fixes #1807
    
    <!-- As a practice, it is recommended to open an issue to have
    discussions on the proposed pull request.
    This makes it easier for the community to keep track of what is being
    developed or added, and if a given feature
    is demanded by more than one party. -->
    
    ## Type of change
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have made corresponding changes to the documentation (not
    needed)
    - [x] My changes generate no new warnings
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    ---------
    
    Signed-off-by: samibouziri <79418773+samibouziri@users.noreply.github.com>
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>

[33mcommit 2cf28672de51f2ecfdd0f0c47fdddd26d0cc26cf[m
Author: Louis LE LAY <le.lay.louis@gmail.com>
Date:   Thu Feb 6 11:31:43 2025 -0500

    Fixes incorrect local documentation preview path in xdg-open command (#1776)
    
    # Description
    
    This PR fixes the `xdg-open` command in the documentation that instructs
    users on how to preview the documentation locally. Previously, the
    command pointed to `docs/_build/html/index.html`, which is incorrect.
    The correct path is now `docs/_build/current/index.html`.
    
    ## Type of change
    
    - Bug fix (non-breaking change which fixes an issue)
    - This change requires a documentation update
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there

[33mcommit edb33d3d3930f3e13bcca350101f665d381df41e[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Feb 6 03:32:43 2025 -0500

    Updates EULA link and spelling in documentation (#1795)
    
    # Description
    
    Documentation only change to change spelling of Nvidia to NVIDIA and
    update the EULA link.

[33mcommit 01c8f25f90582acad6cc46c4d86597ebafadee49[m
Author: Jack Zeng <92033060+Jackkert@users.noreply.github.com>
Date:   Sun Feb 2 05:31:29 2025 +0100

    Fixes timestamp of com and link buffers when writing articulation pose to sim (#1765)
    
    # Description
    
    This PR is linked to https://github.com/isaac-sim/IsaacLab/issues/1756.
    In short, with the recent deprecation of the `body_state_w` variable,
    and then the removal of the deprecation, there are now 3 ways to get the
    states from the `ArticulationData` class: `body_state_w`,
    `body_com_state_w` and `body_link_state_w`. Commit
    999c1e9ab93acb07c0e31acca7bfe796ebfd3ab4 removed the deprecation,
    removing any `write_root_com.*` and `write_root_link.*` calls and
    therefore not updating the `body_com_state_w` and `body_link_state_w`
    until the next physics step. This caused any use of the
    `body_com_state_w` and `body_link_state_w` buffers to be 1 step behind
    after an environment is reset.
    
    Fixes #1762
    
    This PR updates the timestep of the `body_com_state_w` and
    `body_link_state_w` buffers to -1 in the `write_root_pose_to_sim`
    function in the `Articulation` class so that they update correctly. It
    allows for use of all 3 buffers instead of only `body_state_w`.
    
    ## Type of change
    
    - Breaking change (fix or feature that would cause existing
    functionality to not work as expected)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    ---------
    
    Signed-off-by: Jack Zeng <92033060+Jackkert@users.noreply.github.com>
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>

[33mcommit 70cce9e1adc76574248582af2de4911f8d30e79f[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Fri Jan 31 22:46:17 2025 -0500

    Adds RSL RL installation instructions for pip (#1770)
    
    # Description
    
    Due to limitations from PyPI, all dependency packages have to be
    published pip packages.
    Unfortunately, RSL RL currently does not have a pip package, so we could
    not include it into the Isaac Lab package.
    This PR adds the instructions to install RSL RL separately from pip when
    using pip installed Isaac Lab.
    
    ## Type of change
    
    - This change requires a documentation update
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 8fbae844c1c4dad3474bb95e3234d14a02ead300[m
Author: robotsfan <fanziqi614@gmail.com>
Date:   Sat Feb 1 05:34:17 2025 +0800

    Updates the script path in the document (#1766)
    
    # Description
    
    Update the script path in the document
    
    Fixes #1762
    
    ## Type of change
    
    - Bug fix (non-breaking change which fixes an issue)
    - This change requires a documentation update
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there

[33mcommit 157a19e29b0dcafd87abbdad6f2e9428df3f09be[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Jan 30 17:29:47 2025 -0500

    Updates Isaac Sim doc links (#266)
    
    # Description
    
    Isaac Sim 4.5 documentation moved to a new URL, updating links for the
    Isaac Sim docs to new URLs.
    Additionally, we will be updating VERSION for every commit to the repo,
    so this change updates the documentation parsing to use only the major,
    minor, and patch versions from VERSION.

[33mcommit 239537600796c7176ffdd58cbe6d980cd11343bc[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Jan 30 15:05:12 2025 -0500

    Updates document around installation and Hub (#264)
    
    # Description
    
    Documentation only change for installation and Hub pages.

[33mcommit 51731eb16ef1d196ea68db8347ac163803da7593[m
Author: Michael Gussert <michael@gussert.com>
Date:   Thu Jan 30 10:17:16 2025 -0800

    Fixes example commands in installation docs (#263)
    
    forgot to change the actual code for the user that loads the ant example
    after isnstallation
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [X] I have made corresponding changes to the documentation
    - [X] My changes generate no new warnings
    - [X] I have added tests that prove my fix is effective or that my
    feature works
    - [X] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [X] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 18e451159064e4d771ab874b8d4653488fa1d564[m
Author: lgulich <22480644+lgulich@users.noreply.github.com>
Date:   Thu Jan 30 19:12:32 2025 +0100

    Fixes action plotting import (#258)
    
    Without this fix I get the following error when enabling the action plot
    in the IsaacSim GUI:
    
    ```
    2025-01-28 08:28:06 [594,911ms] [Error] [omni.ui.python] NameError: name 'isaacsim' is not defined
    
    At:
      IsaacLab/source/isaaclab/isaaclab/ui/widgets/line_plot.py(501): _build_filter_frame
      IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py(508): render
      IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py(193): step
      venv/lib/python3.10/site-packages/gymnasium/core.py(322): step
      venv/lib/python3.10/site-packages/gymnasium/wrappers/common.py(393): step
      IsaacLab/scripts/environments/inference_agent.py(104): main
      IsaacLab/scripts/environments/inference_agent.py(110): <module>
    ```
    
    - Bug fix (non-breaking change which fixes an issue)
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there

[33mcommit 552d6c282578f996b5d06661327371bb556154cd[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Tue Jan 28 22:12:12 2025 -0800

    Adds release notes and asset caching to docs (#261)
    
    # Description
    
    This is a documentation only change that includes:
    
    * Add new release notes page to track release notes for all previous
    releases
    * Add simulation stability section to troubleshooting page with OVD
    instructions and link to omni physics guide
    * Add documentation around asset caching using Hub
    * Fixes typing issue for documentation building

[33mcommit 4154de3efc344ff70ac1fefee097ce4d06a16731[m
Author: Michael Gussert <michael@gussert.com>
Date:   Tue Jan 28 15:46:15 2025 -0800

    Updates task workflow and other documentation pages (#255)
    
    ## Checklist
    
    - [X] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [X] I have made corresponding changes to the documentation
    - [X] My changes generate no new warnings
    - [X] I have added tests that prove my fix is effective or that my
    feature works
    - [X] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [X] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    ---------
    
    Signed-off-by: Michael Gussert <michael@gussert.com>
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>

[33mcommit e265dccb9f36b4879aa26c5f670216b6bce61c0b[m
Author: peterd-NV <peterd@nvidia.com>
Date:   Tue Jan 28 10:28:47 2025 -0500

    Fixes env unwrapped error in annotate demos (#260)
    
    Fixes # (issue)
    
    Fix regression of env error due to not using env unwrapped in annotation
    script
    
    <!-- As a practice, it is recommended to open an issue to have
    discussions on the proposed pull request.
    This makes it easier for the community to keep track of what is being
    developed or added, and if a given feature
    is demanded by more than one party. -->
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    - New feature (non-breaking change which adds functionality)
    - Breaking change (fix or feature that would cause existing
    functionality to not work as expected)
    - This change requires a documentation update
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit b59872efd2e272a3c9a77319022db6686e6d52c9[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Mon Jan 27 19:00:26 2025 -0800

    Updates asset path and version for release (#256)
    
    Updates the asset path from staging to production to prepare for the
    release.
    VERSION has also been bumped from 1.4.0 to 2.0.0.
    In addition, updating SKRL dependency to latest 1.4.1 release.

[33mcommit cb771e252889f68df05f5e2ab41b34eba1a8b2d2[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Sun Jan 26 23:09:08 2025 -0800

    Updates script links in sensor docs (#254)
    
    # Description
    
    This is a document only change for updating script links in sensor docs

[33mcommit ca2a36e0785b587c5d9579788de544bec6fb684f[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Sun Jan 26 22:29:46 2025 -0800

    Adds option to filter collisions and real-time playback (#253)
    
    This change introduces a new setting in InteractiveSceneCfg to allow
    specifying whether collision filtering across environments is desired.
    Note that when using the direct workflow, this option will only be
    applied automatically when replicate_physics is also enabled in the
    config. If replicate_physics is not enabled, a warning will appear in
    the logs to prompt users to make a call to scene.filter_collisions().
    This is required because collision filtering happens as part of the
    physics replication process.
    
    In addition, real-time playback support is extended for all RL library
    play.py scripts, allowing real-time replay when possible for
    inferencing.
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    - New feature (non-breaking change which adds functionality)
    - This change requires a documentation update
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 6d9cfc494b4bd2873c2bd8a5422d81981650cce3[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Wed Jan 22 19:48:03 2025 -0500

    Updates warning logging when writing joint limits (#252)
    
    In `omni.isaac.lab.assets.Articulation.write_joint_limits_to_sim`, we
    previously added a check for if default joint positions exceed the new
    limits being set. When this is True, we log a warning message to
    indicate that the default joint positions will be clipped to be within
    the range of the new limits. However, the warning message can become
    overly verbose in a randomization setting where this API is called on
    every environment reset. We now default to only writing the message to
    info level logging if called within randomization, and expose a
    parameter that can be used to choose the logging level desired.
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    - New feature (non-breaking change which adds functionality)
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit f4548020cca64d23bb9395267b019a3132f26515[m
Author: matthewtrepte <mtrepte@nvidia.com>
Date:   Wed Jan 22 16:41:00 2025 -0800

    Fixes if statement conditions in isaaclab install/launch script (#251)
    
    # Description
    
    <!--
    Thank you for your interest in sending a pull request. Please make sure
    to check the contribution guidelines.
    
    Link:
    https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html
    -->
    
    Adding "-gt 0" to complete 2 if statement conditions in the launch
    script.
    
    Change is taken out of - https://github.com/isaac-sim/IsaacLab/pull/1622
    by steple.
    
    <!-- As a practice, it is recommended to open an issue to have
    discussions on the proposed pull request.
    This makes it easier for the community to keep track of what is being
    developed or added, and if a given feature
    is demanded by more than one party. -->
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Signed-off-by: matthewtrepte <mtrepte@nvidia.com>

[33mcommit 51cfcff5c5f21350991334e0dfd33911ceab1dee[m
Author: oahmednv <oahmed@Nvidia.com>
Date:   Wed Jan 22 14:16:17 2025 -0500

    Fixes typo in /physics/autoPopupSimulationOutputWindow setting (#249)
    
    …aclab.sim.SimulationContext
    
    fixed typo in /physics/autoPopupSimulationOutputWindow setting in
    isaaclab.sim.SimulationContext
    
    - Bug fix (non-breaking change which fixes an issue)
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [x] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 5a2d2687f78a917f2191e09651fff78be7d38193[m
Author: Toni-SM <aserranomuno@nvidia.com>
Date:   Wed Jan 22 12:58:15 2025 -0500

    Updates RL libraries' Training Performance table (#250)
    
    # Description
    
    This PR updates the RL libraries' Training Performance table using same
    GPU (NVIDIA GeForce RTX 4090)
    
    ## Screenshots
    
    Agent cfg (to have the same parameters) and training scripts
    modifications
    ![Screenshot from 2025-01-22
    11-06-15](https://github.com/user-attachments/assets/c9a6b1ec-ea4d-4a04-82aa-3c08a528ef24)
    
    Tensorboard logs
    ![Screenshot from 2025-01-22
    11-24-05](https://github.com/user-attachments/assets/6634500c-4f6b-47e7-adc3-fbbf9f63f3ee)
    
    Terminal logs
    rl_games
    ![Screenshot from 2025-01-22
    11-20-13](https://github.com/user-attachments/assets/8e750db1-79bc-486d-a2b5-661725d44152)
    skrl
    ![Screenshot from 2025-01-22
    11-10-16](https://github.com/user-attachments/assets/be3c8c5f-51c6-4fa8-a24d-f915d12f5c7e)
    rsl_rl
    ![Screenshot from 2025-01-22
    11-06-43](https://github.com/user-attachments/assets/94e7be68-d85c-4fbf-8133-85c4700efcf4)

[33mcommit f1d8fa7261cfcec9aaafa47d728db6f460d796d1[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Tue Jan 21 23:04:33 2025 -0500

    Fixes benchmark script import for RSL RL (#248)
    
    # Description
    
    Fixes benchmark script import for RSL RL
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 1f9437f7f444a867a97e4cbd8a28b4b0990850e9[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Mon Jan 20 23:53:03 2025 -0500

    Updates IMU demo script with renaming changes (#247)
    
    # Description
    
    Updates IMU demo script with renaming changes for Isaac Lab 2.0. Mainly
    `omni.isaac.lab` imports have been renamed to `isaaclab`
    
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 8d9ef6fa2ab1d9393ee93ffbaf71e13d786eb1b2[m
Author: CY Chen <cyc@nvidia.com>
Date:   Mon Jan 20 18:07:32 2025 -0800

    Adds subtask annotation checks in annotate_demos.py (#243)
    
    # Description
    
    Added additional checks for subtask annotations in mimic's
    `annotate_demos.py` to make sure the exported demos are all with the
    valid annotations required for running data generation in mimic.
    
    Add script to merge HDF5 files into one dataset. Enables users to merge
    together files after running annotate_demos.
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    - New feature
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Co-authored-by: Peter Du <peterd@nvidia.com>

[33mcommit 25965b74e42795bed6e4702df4bddd2121196c59[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Mon Jan 20 21:07:21 2025 -0500

    Fixes error in hdf5 loading and hides simulation settings window (#246)
    
    # Description
    
    This change enforces the Isaac Lab extensions to be loaded last to avoid
    conflicts in hdf5 with omniverse extensions. Additionally, we hide the
    Simulation Settings window by default in `SimulationContext` as
    specifying it in the app file does not seem to work. Also adds the
    IsaacLab folder to python path for benchmarking scripts.
    
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 11474763c119bd3597a12b8a7cacb41430a0d142[m
Author: Toni-SM <aserranomuno@nvidia.com>
Date:   Mon Jan 20 14:42:29 2025 -0500

    Adds checkpoint CLI argument to skrl's train script to resume training (#244)
    
    # Description
    
    Add `checkpoint` CLI argument to skrl's train script to resume training.
    It solves https://github.com/isaac-sim/IsaacLab/issues/1635
    
    ## Type of change
    
    - New feature (non-breaking change which adds functionality)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>

[33mcommit cec2918cc1b33867cc5a61ab393c17fce38c013c[m
Author: Toni-SM <aserranomuno@nvidia.com>
Date:   Mon Jan 20 13:04:15 2025 -0500

    Adds a note for AMP training/evaluation in docs (#245)
    
    # Description
    
    Add a note for AMP training/evaluation in docs
    
    ## Screenshots
    
    
    ![image](https://github.com/user-attachments/assets/30db926f-5199-4d6a-bf36-501373744b73)

[33mcommit 952099607d29593601dda48155ecac03ce99b304[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Jan 16 23:27:25 2025 -0500

    Updates and fixes for documentation (#242)
    
    This is a documentation only PR that fixes some minor grammar and
    formatting errors, and some small updates.

[33mcommit 485310213a38d08ec38d120f9f3942bf24219b6e[m
Author: oahmednv <oahmed@Nvidia.com>
Date:   Thu Jan 16 23:11:29 2025 -0500

    Adds a tutorial for policy inference in a prebuilt USD scene (#231)
    
    # Description
    
    This PR adds a tutorial to show how to inference on a trained policy in
    a prebuilt USD scene.
    It includes a script that performs inference of the
    `Isaac-Velocity-Rough-H1-v0` environment in a warehouse scene and an
    accompanying tutorial that walks through the script implementation.
    
    
    ## Type of change
    
    - This change requires a documentation update
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [x] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>

[33mcommit 8ff0b78a0dfae526a9ecc715a9e1f88ab459cee3[m
Author: Toni-SM <aserranomuno@nvidia.com>
Date:   Thu Jan 16 23:09:17 2025 -0500

    Adds humanoid AMP tasks for direct workflow (#227)
    
    # Description
    
    This change adds 3 additional Humanoid AMP tasks:
    
    - Isaac-Humanoid-AMP-Dance-Direct-v0
    - Isaac-Humanoid-AMP-Run-Direct-v0
    - Isaac-Humanoid-AMP-Walk-Direct-v0
    
    In addition, SKRL dependency is updated from 1.3 to 1.4.
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - New feature (non-breaking change which adds functionality)
    - This change requires a documentation update
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Signed-off-by: peterd-NV <peterd@nvidia.com>
    Signed-off-by: Kelly Guo <kellyg@nvidia.com>
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Signed-off-by: Ashwin Varghese Kuruttukulam <123109010+ashwinvkNV@users.noreply.github.com>
    Co-authored-by: peterd-NV <peterd@nvidia.com>
    Co-authored-by: CY Chen <cyc@nvidia.com>
    Co-authored-by: oahmednv <oahmed@Nvidia.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: rwiltz <165190220+rwiltz@users.noreply.github.com>
    Co-authored-by: nv-cupright <92540563+nv-cupright@users.noreply.github.com>
    Co-authored-by: Alexander Poddubny <143108850+nv-apoddubny@users.noreply.github.com>
    Co-authored-by: chengronglai <chengrongl@nvidia.com>
    Co-authored-by: David Hoeller <dhoeller@nvidia.com>
    Co-authored-by: matthewtrepte <mtrepte@nvidia.com>
    Co-authored-by: Ashwin Varghese Kuruttukulam <123109010+ashwinvkNV@users.noreply.github.com>
    Co-authored-by: Karsten Patzwaldt <kpatzwaldt@nvidia.com>

[33mcommit aecf9afdc3c1487cb538135b751595857464ecbe[m
Author: matthewtrepte <mtrepte@nvidia.com>
Date:   Thu Jan 16 20:08:09 2025 -0800

    Adds unit tests for multi tiled cameras  (#241)
    
    <!--
    Thank you for your interest in sending a pull request. Please make sure
    to check the contribution guidelines.
    
    Link:
    https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html
    -->
    
    Adds initial unit tests for multiple tiled cameras, which had several
    bugs/features recently. The unit tests cover existing test cases from
    the test_tiled_camera.py and a few new ones, including initialization,
    groundtruth annotators, different poses, and different resolutions.
    
    <!-- As a practice, it is recommended to open an issue to have
    discussions on the proposed pull request.
    This makes it easier for the community to keep track of what is being
    developed or added, and if a given feature
    is demanded by more than one party. -->
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - New feature (non-breaking change which adds functionality)
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [ ] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>

[33mcommit 118201f7ebb0eac9e5fcd6b5a6d76cde8adbdfce[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Jan 16 21:36:43 2025 -0500

    Updates check_instanceable script and documentation (#240)
    
    # Description
    
    This PR makes a few minor changes:
    
    - Increases GPU buffer dimensions in check_instanceable.py script to
    better support testing with 4096 environments
    - Updates livestream documentation path to point to new Isaac Sim doc
    sections
    - Updates extension template instructions with new renaming changes
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    - This change requires a documentation update
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 911657573765d52260bceef67471cfaf25fccb2f[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Thu Jan 16 21:31:08 2025 -0500

    Fixes modules being loaded when defined in extension.toml file (#237)
    
    # Description
    
    In the extension.toml for the isaaclab_rl and isaaclab_tasks extensions,
    pipapi and module tags are specified for the various RL libraries. These
    tags introduce an import call that loads these packages.
    
    The SKRL package in particular will initialize torch.distributed if
    running distributed training, causing errors if other RL libraries are
    being used, since other libraries will try to initialize
    torch.distributed again in its own code.
    
    This should ideally be fixed from the SKRL side.
    
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 944245a2df423f9d3224b77cd55ad8b8b3f0395b[m
Author: jaczhangnv <jaczhang@nvidia.com>
Date:   Thu Jan 16 15:17:10 2025 -0800

    Improves XR teleop performance (#236)
    
    # Description
    
    - Updated the simulation parameters to improve the XR teleop performance
    
    ## Type of change
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Co-authored-by: Rafael Wiltz <rwiltz@nvidia.com>

[33mcommit 027ba928a777613fd3ab174b2ee70d9c204cd306[m
Author: Toni-SM <aserranomuno@nvidia.com>
Date:   Thu Jan 16 16:07:47 2025 -0500

    Installs toml dependency in Dockerfile.base (#239)
    
    # Description
    
    Install `toml` dependency in docker file. This prevents the
    `ModuleNotFoundError: No module named 'toml'` error when installing
    isaac lab dependencies in the docker build process.

[33mcommit c80cd685be64fa2cdd553a84eede990689bbb608[m
Author: peterd-NV <peterd@nvidia.com>
Date:   Thu Jan 16 14:42:14 2025 -0500

    Adds robomimic to IsaacLab install script (#238)
    
    Add robomimic back to isaaclab.sh install script
    
    Fixes # (issue)
    
    Fixes robomimic not found error when running Mimic tutorial
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 2bdec262f4637363b6d16214921c2f101f31c4dd[m
Author: rwiltz <165190220+rwiltz@users.noreply.github.com>
Date:   Wed Jan 15 17:08:18 2025 -0500

    Fixes env unwrapped issue when using hand tracking teleop (#233)
    
    # Description
    
    Fix use of env vs env unwrapped and set xy_rotation flag for
    handtracking in teleop script
    
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [x] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [x] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>

[33mcommit 4ed3c225dff4490305043fce240e1ed76e812093[m
Author: peterd-NV <peterd@nvidia.com>
Date:   Wed Jan 15 15:29:50 2025 -0500

    Fixes env.unwrapped errors in recorder/replayer scripts (#235)
    
    Set `env` to be `env.unwrapped` during initial environment creation to
    avoid needing to manually specify `env.unwrapped` multiple times later
    in the scripts, which would often lead to one being missed causing an
    error.
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 2101c5551731e7544354e08e037ca1af987edc8d[m
Author: peterd-NV <peterd@nvidia.com>
Date:   Wed Jan 15 14:38:36 2025 -0500

    Adds pre-recorded dataset link to Mimic docs (#232)
    
    # Description
    
    Update Isaac Lab Mimic docs with S3 link to pre-recorded human demo
    dataset.
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Documentation change
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>

[33mcommit 40c8d6cf044ef41ec5a17973d9187255f4df8499[m
Author: Kelly Guo <kellyg@nvidia.com>
Date:   Tue Jan 14 00:18:00 2025 -0500

    Moves asset path from AppLauncher to app files (#228)
    
    In a recent commit, Isaac Lab extensions were added to the app files as
    dependencies in order to support installing Isaac Lab modules from pip.
    However, this introduced a new issue where the assets module may be
    loaded prior to the asset base path setting was set in AppLauncher,
    causing assets to not be found. This causes the first runs of Isaac Lab
    scripts to fail as the carbonite setting we look for in the assets
    script has not been populated yet.
    
    To fix this, this PR moves the definition of the asset path from
    AppLauncher to the app files. This ensures that the carbonite setting is
    set prior to any code being executed.
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [ ] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [x] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->

[33mcommit 8a58a23cdad4dfdb279090306e0762d4f0c98381[m
Author: peterd-NV <peterd@nvidia.com>
Date:   Mon Jan 13 22:27:13 2025 -0500

    Updates Mimic APIs/configs/docs for future dexmimic compatibility (#216)
    
    # Description
    
    Doc and config changes from @karsten-nvidia:
    - Add additional details on custom environments to mimic docs.
    - Update comments in mimic configuration to make it easier telling apart
    what's important.
    - Some minor cleanups in existing docs.
    - Add "common pitfalls" section to docs to guide users how to get
    successful data generation/training
    
    Mimic API and config changes to support forward dexmimic compatibility:
    - Use dictionaries of subtasks in mimic env config; keys are eef_names
    - Mimic Env APIs now use dictionary of eef_names to enable mulit-eef
    support in future
    - Data generation code updated accordingly to use new Mimic env APIs
    
    ## Type of change
    
    <!-- As you go through the list, delete the ones that are not
    applicable. -->
    
    - Bug fix (non-breaking change which fixes an issue)
    
    ## Screenshots
    
    Please attach before and after screenshots of the change if applicable.
    
    <!--
    Example:
    
    | Before | After |
    | ------ | ----- |
    | _gif/png before_ | _gif/png after_ |
    
    To upload images to a PR -- simply drag and drop an image while in edit
    mode and it should upload the image directly. You can then paste that
    source into the above before/after sections.
    -->
    
    ## Checklist
    
    - [x] I have run the [`pre-commit` checks](https://pre-commit.com/) with
    `./isaaclab.sh --format`
    - [x] I have made corresponding changes to the documentation
    - [x] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my
    feature works
    - [ ] I have updated the changelog and the corresponding version in the
    extension's `config/extension.toml` file
    - [ ] I have added my name to the `CONTRIBUTORS.md` or my name already
    exists there
    
    <!--
    As you go through the checklist above, you can mark something as done by
    putting an x character in it
    
    For example,
    - [x] I have done this task
    - [ ] I have not done this task
    -->
    
    ---------
    
    Signed-off-by: peterd-NV <peterd@nvidia.com>
    Signed-off-by: Kelly Guo <kellyg@nvidia.com>
    Signed-off-by: Kelly Guo <kellyguo123@hotmail.com>
    Signed-off-by: Ashwin Varghese Kuruttukulam <123109010+ashwinvkNV@users.noreply.github.com>
    Co-authored-by: CY Chen <cyc@nvidia.com>
    Co-authored-by: oahmednv <oahmed@Nvidia.com>
    Co-authored-by: Toni-SM <aserranomuno@nvidia.com>
    Co-authored-by: Kelly Guo <kellyg@nvidia.com>
    Co-authored-by: Kelly Guo <kellyguo123@hotmail.com>
    Co-authored-by: rwiltz <165190220+rwiltz@users.noreply.github.com>
    Co-authored-by: nv-cupright <92540563+nv-cupright@users.noreply.github.com>
    Co-authored-by: Alexander Poddubny <143108850+nv-apoddubny@users.noreply.github.com>
    Co-authored-by: chengronglai <chengrongl@nvidia.com>
    Co-authored-by: David Hoeller <dhoeller@nvidia.com>
    Co-authored-by: matthewtrepte <mtrepte@nvidia.com>
    Co-authored-by: Ashwin Varghese Kuruttukulam <123109010+ashwinvkNV@users.noreply.github.com>
    Co-authored-by: Karsten Patzwaldt <kpatzwaldt@nvidia.com>
