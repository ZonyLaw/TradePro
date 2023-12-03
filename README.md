# TradePro

This app gives you some insights on potential targets and profits based on current prices. It uses machine learning to capture what trade is expected within a specific timeframe.

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
