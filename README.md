# Semantic
Included files have to do with semantic similarity and NLP's

## Creating a virtual environment
python -m venv nameofenvironment (main folder)
*Replace nameofenvironment with your own environment name*
nameofenvironment\Scripts\activate (Activate virtual environment)
Install required modules from the requirements.txt file
cd to go into the folder of your project
run server with the following command:
python manage.py runserver

## Creating your own Docker Image
*Ensure that you have Docker installed and running*
*Enter the powershell as an administrator*
Enter into the powershell:
1. docker build -t myapp . (replace myapp with what you want to name the image)
2. Create a repository in your Docker hub with THE SAME NAME as your image (In this case myapp)*This should be creatd on your Docker hub Account and not in the powershell*
3. docker tag myapp  [username]/myapp  (Back to the powershell)*Replace username with your username and myapp with your name of the repository*
4. docker login (In the powershell. This grants access to your Docker Hub Account)
5. docker push [username]/myapp:latest (In the powershell. This pushes your local project to your repository)
6. Go to Docker Playground and add a new instance.
7. Thereafter run *docker pull [username]/myapp:latest*
8. Thereafter run *docker run -it [username]/myapp:latest*
