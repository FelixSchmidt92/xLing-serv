# xLingQG-serv
Cross lingual question generation server
```
source venv/bin/activate to active venv
```

# build image and run 
Increase the maximum Memory docker can consume in the settings of docker on your local machine
````
docker build -t xqg .
docker run -p 5003:5003 xqg
````
