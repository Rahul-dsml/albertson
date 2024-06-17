# albertson
image keywords extraction

# Steps
create a requirements.txt
- pip freeze >requirements.txt

upload the project files to github repo

install azure-cli (it will take some time)
- pip install azure-cli

login to azure account
- az login

create an azure container registry
 - az acr create --resource-group <resource-group-name> --name <acr-name> --sku Basic

build the docker image using azure container registry
- az acr build --registry <acr-name> --image myapp:latest --file Dockerfile https://github.com/<your-github-username>/<your-repo>.git

create an app service
 - az appservice plan create --name <app-service-plan> --resource-group <resource-group-name> --sku B1 --is-linux

create a web app that will use your docker image
 - az webapp create --resource-group <resource-group-name> --plan <app-service-plan> --name <web-app-name> --deployment-container-image-name <acr-name>.azurecr.io/<image-name>:<tag>


