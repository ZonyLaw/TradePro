# TradePro


Currently, the app is still under development and provides only one product, USDJPY. While the main structure is in place, it requires cleaning up. 
The app offers insights into potential market direction. It utilises machine learning to predict trading activity within a specific timeframe (ie. 1hr and 4 hr).
[Live Site - TradePro ](https://trade-pro-4909851596e5.herokuapp.com/)

![image](https://github.com/ZonyLaw/TradePro/assets/104385712/f10fa398-f032-4f1f-9521-05175c052b83)


## Table of Contents

- [TradePro](#TradePro)
  - [Table of Contents](#table-of-contents)
    - [Project-Management-Plan](#Project-Management-Plan)
    - [Webpage-Structure-Plan](#Webpage-Structure-Plan)
    - [Appendix](#Appendix)

# Project-Management-Plan

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

**EPIC 9 - Improving performance**

This is improve the accessing of model results using MongoDB as storage. This will overcome issue of writing and reading conflicts when using a .JSON file as storage.
Once this is confirm working, the .JSON method of saving results will be discontinue.

**EPIC 9 - Subscription**

This is to develop a subscription feature where users can signup to the services on this app.

**EPIC 10 - Documentation**

This epic is for all documents related stories and tasks to document the lifecycle of the project development. It aims to record the detailed documentations to all stages of development and necessary information on testing, deploying and running the application.

# Notes:


# Webpage-Structure-Plan

## Features

**Home Page**

*1. Navigation Menu*

*User Story*

``As a developer and user, I can have use the navbar to navigate the website from any device.``
``As a trader, I can view the model results and analysis to get a potential direction of the market trend.``


**Price**
``As a developer and user, I can view the prices for a specific currency.``
``As a developer , I edit, export of a specific currency.``


**models**
``As a user, I can view a report on the projections of market trend.``

**email alert**
``As a user, I receive alert of potential trade coming up.``

**news**
``As a user, I have a view of news coming out for the week.``
``As a user, I have the news model to find out if there are any potential impact on market trends.``


**Report**

*User Story*
``As a trader, I can view the model analysis for specific currency.``


**CRON**
The cron scheduler is run in the app, prices. This is most logical as the new prices will need revised calculations.
The file containing this code is called, updater.py.

<details open>
<summary>Implementation:</summary>


 The Navigation bar contains links for Home, prices, , Register, Login and Contact.
  * Home -> index.html - Visible to all
  * Prices -> show the prices of the ticker - Visible to all
    * export prices - export all prices for the specified ticker - Visible to Admin
    * upload prices  - upload prices for the specified ticker - Visible to Admin
    * update prices  - manual update of a prices for the specified ticker - Visible to Admin
  * Predictions -> show the model results and anslysis - Visible to all
  * Variability -> show the variability of results giving the sensistivity test - Visible to all
  * Account -> view user profile and update details - visible to user
    
  * Login -> login.html - Visible to logged out users
  * Register -> signup.html - Visible to logged out users
  * Logout -> logout.html - Visible to logged in users


**MongoDB**

* To ensure access to the MongoDB, make sure setting for the network access IP is updated. This can change and produce error message relating to SLL.
* Here is some notes on SLL in case this becomes an issue in the future. Please also refer to Udemy.
  * https://slproweb.com/products/Win32OpenSSL.html
  * openssl req -newkey rsa:2048 -new -x509 -days 3650 -nodes -out mongodb-cert.crt -keyout mongodb-cert.key -config "C:\Program Files\OpenSSL-Win64\openssl.cnf"
  * type mongodb-cert.key mongodb-cert.crt > mongodb.pem

# Appendix

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

- python manage.py test tests

### commands to clean the local database

- python manage.py flush

- python manage.py migrate your_app_name zero.....also delete the migration you don't want

## challenges in creating this app

Here are some notes for future referencing.
- the key wasn't set up correctly and this throw an error
- wsgi.py reference to the wrong app name
- user specific timezone rendering needs {% load tz %} in html combined with filtering
{{ price.date|timezone:user_timezone }}; backend needs a conversion process also.
- heroku have a limit of 500mb storage so remove unneccesary packages
- commented out the cloudinary setting as there seem to be issue with staticfile.json. This may be related to version issue or I have not set the cloudinary properly.
- cst refer to wrong reference to dictionary
- it is not recommended that development access production database. If you want to access the PostgresSQL, so Django version 4.1 is used rather the latest version.
- The local development server runs a separate process for the auto-reloader during a cron schedule. You can turn off the auto-reload process by passing the --noreload flag -> python manage.py runserver --noreload
- In heruko, to prevent the cron scheduler runing twice, the environment setting needs to have WEB_CONCURRENCY = 1.


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

## Env file content

import os

os.environ['CLOUDINARY_URL'] = ''
os.environ['DATABASE_URL'] = ''
os.environ['SECRET_KEY'] = ''
os.environ['IG_USERNAME'] = ''
os.environ['IG_PASSWORD'] = ''
os.environ['IG_API_KEY'] =  ''
os.environ['GMAIL_PASSWORD2'] = ''
os.environ['MONGO_URL'] = ''

making static work on heroku
remove this variable in heroku: DISABLE_COLLECTSTATIC = 1
install whitenoise as middleware
ensure the dubug is false



## conda env 

 - Using Python 3.11, as it aligns with the requirements of PostgreSQL.
 - Use conda env, myenv311 and myenv311test.

