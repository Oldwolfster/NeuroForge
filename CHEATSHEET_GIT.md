# Steps to sync to gitHub
git status                              # Check the status of your files
git add .                               # Stage all changes
git commit -m "Your commit message"     # Commit changes with a message
git push origin master                  # Push changes to the 'master' branch on 'origin'
OR
git push                                # If your local master branch is already tracking origin/master, you can simply use
g


---
# Alternate branches
git checkout main
git checkout original_adam_build

---
# Rollback to last commit
git reset --hard HEAD

---
# Steps to create "snapshot in time" for each episode
## freeze progress for an episode, create a branch:
git checkout -b episode_x_yadayada                   # Create and switch to a new branch
git push -u origin episode_x_yadayada                # Push to 'origin' and set upstream


## Switch back to the main branch to continue working
git checkout master                                  # Switch back to 'master' branch

## Update the frozen branch with new code.
Commit and upload on master branch
git checkout episode004_From_Zero_To_Regression_Hero
git reset --hard master
git push -f origin episode004_From_Zero_To_Regression_Hero


## View local branches
git branch                                           # List all local branches
(NOTE: highlights the current one with an asterisk (*).)
    
## View remote branches
git branch -r                                        # List all remote branches

## check if your local branches have upstream branches
git branch -vv

## View differences between branches
git diff master feature_branch

---
# Steps to view old version of file
git log --follow -- src/Metrics.py                   # View commit history for 'src/Metrics.py'


## show commits where file has changed with nice formmating of date
git log --follow --pretty=format:"%h %ad | %s" --date=short src/Metrics.py


## View differences
git diff <commit_hash> <path_to_file>                # Show differences between a commit and working directory
Example:
git diff f8db2ee798a22cad26fd872170e976276a37bd15 /src/Metrics.py

## View differences between 2 commits
git diff <commit_hash1> <commit_hash2> <path_to_file>


## restore old version to different file.
git show <commit_hash>:<path_to_file> > <path_to_new_file>
Example:
git show f8db2ee:src/Metrics.py > src/Metrics_old_version.py



##TODO List
DONE:  In convergence metrics is fcked and so is perfect accuracy  XOR with seed 547298 in 88, but more perfect epochs first
DONE: Loss functions
get perceptron working