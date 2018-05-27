module.exports = (() => {
    const fs = require("fs");
    const path = require("path");
    const express = require('express');
    const router = express.Router();

    router.get('/',function(req,res){
        res.sendFile(path.resolve(__dirname,"./views/home.html"));
    })

    router.get('/sent',function(req,res){
        res.sendFile(path.resolve(__dirname,"./views/sent.html"));
    })

    router.get('/output',function(req,res){
        res.sendFile(path.resolve(__dirname,"./views/output.html"));
    })
  
  
    router.get('/img/:name',function(req,res){
  
      res.sendFile(path.resolve(__dirname,"../../public/img/"+req.params.name));
  
  
    })
  
    return router;
  })();