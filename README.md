# TradePro

This app gives you some insights on potential targets and profits based on current prices. It uses machine learning to capture what trade is expected within a specific timeframe.

Currently it is still underdevelopment and only provide one product. The main structure is there but will need cleaning up.

## Table of Contents

- [TradePro](#TradePro)
  - [Table of Contents](#table-of-contents)

# User-Experience-Design

## The-Strategy-Plan

### Site-Goals

The primary goal is to provide traders with some insights if the current trend is a buy or sell trade. From this it will use machine learning to give the probability on how big the profit and loss could be within a specific timeframe.

To achive this, the site will collect price data from a specific ticker. The data is accessiable only for admin. There may be possible consideration to also share these prices to the users.

### Agile Planning

This project was developed through applying agile methodologies by delivering small features in incremental sprints.

All projects were assigned to epics, which were broken into small tasks and prioritized as must have, should have, could have. "Must have" stories were completed first before "should haves", and the last "could haves". To ensure all core requirements completed first gives a complete product, with the nice to have features being added if the time frame allows.

The User Story was created using github projects and can be located [here](https://github.com/users/ZonyLaw/projects/3) and can be viewed for more information on the project kanban board. All stories except documentation tasks have a full set of acceptance criteria to mark the story is completed.

#### Epics

The project had 8 main Epics (milestones):

**EPIC 1 - Base Setup**

This step is to ensure all key packages are installed and the general folder structure established with app folders to contain the model and functions.
Also, connection with cloudinary and heruko is established before continuing to avoid runtime errors when the app is ready ready to be rollout.

**EPIC 2 - Small Pages**

Instead of creating epics for tiny features, all the small pages were included to this epic.

**EPIC 3 - Authentication Epic**

The authentication epic is for all stories related to the registration, login and authorization of views. It allows admin and user to have have the CRUD ability of the tables to be created.

**EPIC 4 - Ticker**

The ticker epic is a list of tickers the user can access for insights with full CRUD capabiility.

**EPIC 5 – Prices**

The price epic will record hourly prices (ie. open, close, high, low, volume) for tickers via an API endpoint.

**EPIC 6 – Reviews [for future development]**

The reviews epic is to allow users to put comments on the models or trade ideas to help improve the website.

**EPIC 7 - Developing model to predict PL**

This is the development of a model using machine learning based on historical data. This is done separate from this app and under project NeuralPricing.

**EPIC 8 - Integrating model with app to display results**

This epic is integrating the model into this app so it could produce hourly results giving insights what type of trades is at play (ie. buy or sell) and the probability for a range of profit and losses.
It will also suggest some targets and stop loss. This is to help a trader to make better decision on how to place the trade.

**EPIC 9 - Subscription**

This is to develop a subscription feature where users can signup to the services on this app.

**EPIC 10 - Documentation**

This epic is for all documents related stories and tasks to document the lifecycle of the project development. It aims to record the detailed documentations to all stages of development and necessary information on testing, deploying and running the application.

# Notes:

## Key commands

- pip3 install django gunicorn

- pip3 install dj_database_url psycopg2

- pip3 install dj3-cloudinary-storage

- pip3 freeze > requirements.txt

- creating the django admin

- django-admin startproject <project name>

- python3 manage.py startapp <app name>

- python manage.py makemigrations

- python manage.py migrate - this builds some default tables and you can access them via admin route.

- python manage.py createsuperuser

- running the app

- python manage.py runserver

## commands to clean the local database

- python manage.py flush

- python manage.py migrate your_app_name zero.....also delete the migration you don't want

### challenges in creating this app

Here are some notes for future referencing.
- the key wasn't set up correctly and this throw an error
- wsgi.py reference to the wrong app name
- user specific timezone rendering needs {% load tz %} in html combined with filtering
{{ price.date|timezone:user_timezone }}; backend needs a conversion process also.
- heroku have a limit of 500mb storage so remove unneccesary packages
- commented out the cloudinary setting as there seem to be issue with staticfile.json. This may be related to version issue or I have not set the cloudinary properly.
- cst refer to wrong reference to dictionary
- it is not recommended that development access production database. If you want to access the PostgresSQL, so Django version 4.1 is used rather the latest version.


#### Using email as login

- To convert from username to email as login, the database, migration files, and cach need to be deleted.

- here are some key commands:

  * clear all pycache
    find . -name "*.pyc" -exec rm -f {} \;

  * clean up any files after using "git restore ."
    git clean -fdx

- Ensure any custom made apps link to the user need to be deleted before following the steps found at this website:
https://pypi.org/project/django-use-email-as-username/

- A package is installed for using email instead of username (pacakge installation: pip install django-use-email-as-username)

### Content in the env file if this gets deleted

import os

os.environ['CLOUDINARY_URL'] = ''
os.environ['DATABASE_URL'] = ''
os.environ['SECRET_KEY'] = ''
os.environ['IG_USERNAME'] = ''
os.environ['IG_PASSWORD'] = ''
os.environ['IG_API_KEY'] =  ''


### conda env 

working env is myenv311 and myenv311test

### task

need to find ways to store the results and records