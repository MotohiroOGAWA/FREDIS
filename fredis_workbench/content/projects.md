# Workbench projects

Use **Home → Open / Create Project** to choose a research directory. An existing FREDIS project is reopened; otherwise Workbench initializes one in that directory. The project does not have to contain the FREDIS source code. The application location and Python environment continue to use the existing FREDIS preferences.

The last project is reopened with the workspace. Recent projects are available on Home. **Close Project** returns to the ordinary Workbench without removing any project files.

## Pick up where you left off

Settings in Data Preparation, Training, Prediction, registered Reaction Patterns, and Home's quick forms are saved automatically after a short pause. Switching projects saves the current forms and restores the selected project's forms. Visual pattern edits enter the saved configuration after **Add Reaction Pattern to Set**; unfinished visual-editor operations are not project snapshots.

New projects suggest separate output directories under `runs/<workflow>-<timestamp>-<suffix>`. You can choose another location. Relative paths in workflow settings retain their existing meaning: they resolve against the FREDIS application directory, not the project directory.

## Keep reproducible settings

Use **Save Snapshot** on Home for a named checkpoint of your settings. Workbench also records configuration loads/exports and the settings submitted to data preparation, training, single prediction, and batch prediction. Run snapshots link to their job records. Snapshots store the configuration values and file references as they were recorded, including file size and modification time when available.

**Restore Settings** fills the appropriate form and opens it for review. It does not execute anything. Restoring a history entry chooses a fresh output directory and turns off overwrite flags. **Reuse Settings** on a job uses its associated configuration. The latest automatically saved forms, by contrast, retain their chosen output directories.

**Compare Previous** shows parameter changes against the preceding snapshot for the same workflow. **View JSON** opens the stored record. History can be searched by label, workflow, or source configuration path.

## Find data, models and outputs

Files referenced by form settings and run configurations appear in **Tracked files**, including external files. Run logs, output directories, and common output artifacts are recorded as well. **Discover Project Files** finds existing datasets, checkpoints, JSON settings and notes in the directory (up to 500 files, 250 directories and six directory levels; hidden folders and symbolic links are skipped). **Track Files** adds other files, such as experiment notes or reference spectra. Favorite frequently used files, filter by workflow, copy paths, reveal files in Explorer, or restore a file's associated settings.

Missing files and files modified since their last recorded metadata are marked. **Refresh** checks the current paths; Home also refreshes periodically. File contents are not copied or hashed. The project history identifies which paths and settings were used; it does not freeze external datasets or model checkpoints.

## Runs and persistence

Recent Jobs and Training Jobs show the active project's runs. Switching projects does not move an already-started job: its completion status and logs continue to belong to its original project. Existing Workbench jobs from before using projects remain in ordinary workspace history and are available after closing the project.

Project records are stored under:

```
project-directory/
  .fredis/
    project.json              # Project identity, notes, history index, file references
    drafts.json               # Most recently saved forms
    configurations/<id>.json  # Configuration snapshots
    jobs/<id>.json            # Run status, command and associated configuration
    jobs/<id>.log             # Run log
    jobs/<id>.result.json     # Single-prediction result, when available
  runs/                       # Suggested output location
```

Keep `.fredis` when backing up a project. Paths to input and output files are absolute references; moving a project or an external dataset does not rewrite old history. Damaged or unsupported project metadata produces an error instead of being silently replaced.
