# albertson
image keywords extraction

# Steps
create a requirements.txt
- pip list >requirements.txt
upload the project files to github repo

install azure-cli (it will take some time)
- pip install azure-cli

login to azure account
- az login

build the docker image using azure container registry
- az acr build --registry <acr-name> --image myapp:latest --file Dockerfile https://github.com/<your-github-username>/<your-repo>.git



