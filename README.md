# The Tensor Clan - Backend - Heroku

FrontEnd: https://github.com/extensive-vision-ai/thetensorclan-web

This Repository contains the files used for the Heroku backend, since AWS had some file size limitations

## Run 🎬

```shell script
flask run --host=0.0.0.0
```

## Deploy 🚀

```shell script
heroku git:remote -a thetensorclan-backend
git commit -m "update"
git push heroku master
```

## BuildPacks

```shell script
heroku buildpacks:set heroku/python
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks
# Should show apt first, then python
```
