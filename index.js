"use strict"

const path = require('path');
const express  = require('express');
const app      = express();
const session = require('express-session');
const port     = 3000||process.env.MONGODB_URI ;
const index    = require('./router/index')
const configDB = require('./config/dbinit.js');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies
app.use(session({
    secret: 'mrboombaandstarinsane',
    resave: true,
    saveUninitialized: true
}));

const fileUpload = require('express-fileupload');

mongoose.connect(configDB.url);
const cors = require('cors');
app.use(fileUpload());
app.use(cors());
app.use(express.static(path.join(__dirname+ '/public/')));
app.use('/',index);


app.listen(port,function(){
	console.log('Magic is happend on port '+port);
})

