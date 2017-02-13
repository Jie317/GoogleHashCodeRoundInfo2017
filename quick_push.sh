#! /bin/bash

# Script to samplify the commit and push steps. All the arguments are concatenated as the commmit message.

git add -A
if [ $# -eq 0 ]; then
	echo "Warning: Better to leave a message to this commit. Using default message: update"
	echo ""
	git commit -m "Update"
else
	git commit -m "$*"
fi
git push origin master


# Other tips
# 1 ssh login https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/ then run: ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

