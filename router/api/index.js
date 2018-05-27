module.exports = (() => {
  const fs = require("fs");
  const path = require("path");
  const express = require("express");
  const router = express.Router();
  const fileUpload = require("express-fileupload");
  const user = require("../../models/user");
  var PythonShell = require("python-shell");
  var request = require("sync-request");
  var nodemailer = require('nodemailer');

  const csv = require("csvtojson");

  var options = {
    mode: "text",
    pythonOptions: ["-u"], // get print results in real-time
    scriptPath: path.join(__dirname, "../../python/"),
    args: [path.join(__dirname, "../../video/")]
  };

  const files = fs.readdirSync(__dirname);

  var crypto = require("crypto");

  var transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: 'mrboomba161@gmail.com',
      pass: ''
    }
  });

  //function code taken from http://blog.tompawlak.org/how-to-generate-random-values-nodejs-javascript
  function randomValueHex(len) {
    return crypto
      .randomBytes(Math.ceil(len / 2))
      .toString("hex") // convert to hexadecimal format
      .slice(0, len)
      .toUpperCase(); // return required number of characters
  }

  router.post("/output",function(req,res){
    console.log(req.body.code)
    if(req.body.code){
      user.findOne({code:req.body.code},function(err,users){
        if(err){
          return res.status(500).json({message:"fail"});
        }
        if(!users){
          return res.status(500).json({message:"code not found"});
        }
        else{
        req.session.code = users.code
        return res.status(200).json({message:"success"});
        }
      })
    }
    else{
      return res.status(500).json({message:"no code"});
    }
  })

  router.get("/output",function(req,res){
    if(req.session.code){
      user.findOne({code:req.session.code},function(err,users){
        if(users.location.length>0){
          var array =[];
          array.push(users.location[0]);
          for(var i=0;i<users.direction.length;i++){
            if(array[array.length-1]!=users.direction[i]){
                array.push(users.direction[i])
            }
          }
          var response = {location:users.location,
                          direction:array,
                          locationStr:users.locationString}
          return res.status(200).json(response);
        }else{
          return res.status(500).json({message:"not finish"});
        }
      })
    }
    else{
      return res.status(500).json({message:"no code"});
    }
  })

  router.post("/upload", function(req, res) {
    console.log("UPLOAD!!");
    console.log(req.files.foo);

    if (!req.files) return res.status(400).send("No files were uploaded.");

    // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
    let file = req.files.foo;
    let fileName = file.name;
    let splitArray = fileName.split(".");
    let ext = splitArray[splitArray.length - 1];

    let email = req.body.email;
    let loc = {
      lat: req.body.lat,
      long: req.body.long
    };
    let heading = req.body.heading;

    console.log(req.body);
    var code = randomValueHex(4);
    var usr = {
      email: email,
      filename: code + "." + ext,
      loc: loc,
      code: code,
      heading: heading
    };

    user.create(usr, function(err, newUser) {
      if (err) {
        console.log(err);

        return res.status(500).send(err);
      }
      // Use the mv() method to place the file somewhere on your server
      file.mv(path.join(__dirname, "../../video/" + newUser.filename), function(
        err
      ) {
        if (err) {
          console.log(err);
          return res.status(500).send(err);
        }
        options.args.push(newUser.filename);
        PythonShell.run("data_prepare.py", options, function(err, results) {
          if (err) console.log(err);
          var options2 = {
            mode: "text",
            pythonOptions: ["-u"], // get print results in real-time
            scriptPath: path.join(__dirname, "../../KittiSeg/"),
            args: [
              "--input_image",
              path.join(__dirname, "../../result/" + newUser.code + "/")
            ]
          };
          PythonShell.run("demo.py", options2, function(error, results1) {
            if (error) console.log(error);
            // results is an array consisting of messages collected during execution
            console.log(results1);
            var options3 = {
              mode: "text",
              pythonOptions: ["-u"], // get print results in real-time
              scriptPath: path.join(__dirname, "../../python/"),
              args: [path.join(__dirname, "../../python/"), newUser.code]
            };
            PythonShell.run("predict-multiclass.py", options3, function(
              errors,
              results2
            ) {
              if (errors) console.log(errors);
              // var options4 = {
              //   mode: "text",
              //   pythonOptions: ["-u"], // get print results in real-time
              //   scriptPath: path.join(__dirname, "../../python/"),
              //   args: [path.join(__dirname, "../../python/"), newUser.filename]
              // };
              // PythonShell.run("test.py", options4 , function(error3,results3){
              //   if(error3) console.log(error3);
              //   var options5 = {
              //     mode: "text",
              //     pythonOptions: ["-u"], // get print results in real-time
              //     scriptPath: path.join(__dirname, "../../python/"),
              //     args: [path.join(__dirname, "../../python/"), newUser.code]
              //   };
              //   PythonShell.run("makeCSV.py", options5,function(error4){
              //       if(error4)console.log(error4);
                    // findLocation(newUser,function(finaluser){
                      var mailOptions = {
                        from: 'mrboomba161@gmail.com',
                        to: newUser.email,
                        subject: 'Result from DVA',
                        text: 'We already finish your dashcam analytic for create driving route. Please note this code and put in our website : '+newUser.code
                      };
                      transporter.sendMail(mailOptions, function(error, info){
                        if (error) {
                          console.log(error);
                        } else {
                          console.log('Email sent: ' + info.response);
                        }
                      });
                      
                    });
                });
              });
            // });
          // });
        // });
        
        res.status(200).json({ status: "success" });
      });
    });
  });


  function findLocation(usr,callback) {
    var direction = [];
    var distance = [];
    var loc = [];
    // var heading = 273.059470544951;  //heading straigt BBBB
    // var heading = 1.6252514487809515;  //heading straigt AAAA
    // var initstr = "14.063568,100.61792"  //intloc straigt BBBB
    // var initstr = "14.061834,100.631607"  //intloc straigt AAAA
    var initstr = usr.loc.lat+","+usr.loc.long
    var heading = usr.heading
    var code =usr.code;

    var response = request('GET', "https://roads.googleapis.com/v1/snapToRoads?path="+initstr+"&key=AIzaSyDNo9QiRn5XSKq6Nyt7h9qBzjSRns8pUhU"); // Print the google web page.
    loc.push({ lat: JSON.parse(response.body).snappedPoints[0].location.latitude, long: JSON.parse(response.body).snappedPoints[0].location.longitude });
    console.log(loc);
    
    var csvFilePath = path.join(__dirname, "../../result/"+code+"/"+code+".csv");
    csv()
      .fromFile(csvFilePath)
      .on("json", jsonObj => {
        direction.push(jsonObj.direction);
      })
      .on("done", error => {
        csvFilePath = path.join(__dirname, "../../result/"+code+"/"+code+"_vel.csv");
        csv()
          .fromFile(csvFilePath)
          .on("json", jsonObj => {
            distance.push(jsonObj.velocity * 0.277778);
          })
          .on("done", error => {
            var locTmp = loc[0];
            var str = "";
              var strTmp = "";
            for (var i = 0; i < direction.length; i++) {
              
              if(distance[i]==0)
              continue;
              if (direction[i] == 0) {
                heading-=30
                if(heading<=0){
                  heading = 360+heading;
                }
                // for (var j = 0, k = heading - 90; j < 90; j++, k++) {
                //   if (k >= 360) {
                //     k = 1;
                //   }
                //   locTmp = findNewPoint(loc[loc.length - 1],k,distance[i]);
                //   str += locTmp.lat+","+locTmp.long+"|";
                // }
                locTmp = findNewPoint(loc[loc.length - 1],heading,distance[i]);
                strTmp = str
                strTmp += locTmp.lat+","+locTmp.long+"|";
                strTmp = strTmp.slice(0, -1);

                var response = request('GET', "https://roads.googleapis.com/v1/snapToRoads?path="+strTmp+"&key=AIzaSyDNo9QiRn5XSKq6Nyt7h9qBzjSRns8pUhU");
                locTmp ={lat:JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.latitude,long: JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.longitude}
                
                if (loc[loc.length - 1].lat != locTmp.long &&loc[loc.length - 1].long != locTmp.long){
                  heading = bearingTo(loc[loc.length-1],locTmp)
                  loc.push(locTmp);
                  str += locTmp.lat+","+locTmp.long+"|";
                }

              } else if (direction[i] == 1) {
                heading+=30
                if(heading>=360){
                  heading = heading-360;
                }
                locTmp = findNewPoint(loc[loc.length - 1],heading,distance[i]);
                strTmp = str
                strTmp += locTmp.lat+","+locTmp.long+"|";
                strTmp = strTmp.slice(0, -1);
                locTmp ={lat:JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.latitude,long: JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.longitude}
                if (loc[loc.length - 1].lat != locTmp.long &&loc[loc.length - 1].long != locTmp.long){
                  heading = bearingTo(loc[loc.length-1],locTmp)
                  loc.push(locTmp);
                  str += locTmp.lat+","+locTmp.long+"|";
                  
                }
                
                var response = request('GET', "https://roads.googleapis.com/v1/snapToRoads?path="+strTmp+"&key=AIzaSyDNo9QiRn5XSKq6Nyt7h9qBzjSRns8pUhU");
              } else {
                locTmp = findNewPoint(loc[loc.length - 1],heading,distance[i]);
                strTmp = str
                strTmp += locTmp.lat+","+locTmp.long+"|";
                strTmp = strTmp.slice(0, -1);
                console.log(strTmp);
                
                var response = request('GET', "https://roads.googleapis.com/v1/snapToRoads?path="+strTmp+"&key=AIzaSyDNo9QiRn5XSKq6Nyt7h9qBzjSRns8pUhU");
                locTmp ={lat:JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.latitude,long: JSON.parse(response.body).snappedPoints[JSON.parse(response.body).snappedPoints.length-1].location.longitude}
                if (loc[loc.length - 1].lat != locTmp.long &&loc[loc.length - 1].long != locTmp.long){
                  heading = bearingTo(loc[loc.length-1],locTmp)
                  loc.push(locTmp);
                  str += locTmp.lat+","+locTmp.long+"|";
                }
                
              }
              console.log(str)
              console.log(loc.length);
              
              if(loc.length === 98){
                strr = str.split("|");
                str = "";
                var tmpArray =[];
                for (var k = 1; k < loc.length; k+=2) {
                  tmpArray.push(loc[k])
                }
                for(var k = 0;k<98;k+=2){
                  str+= strr[k]+"|"
                }
                loc = tmpArray
              }
            }

            loc.forEach(element => {
              console.log(element.lat + "," + element.long + " | ");
            });
            usr.location = loc;
            // usr.code = "BBBB";
            // usr.filename = "BBBB.mp4"
            usr.direction = direction;
            usr.locationString = str.slice(0, -1);
            usr.save(function (err){
              if(err) throw err
            })
            return callback(usr);
          });
      });
  }

  function toDegrees(num) {
    return num * 180 / Math.PI;
  }

  function toRadians(num) {
    return num * Math.PI / 180;
  }

  function bearingTo(loc1, loc2) {
    var φ1 = toRadians(loc1.lat),
      φ2 = toRadians(loc2.lat);
    var Δλ = toRadians(loc2.long - loc1.long);
    var y = Math.sin(Δλ) * Math.cos(φ2);
    var x =
      Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
    var θ = Math.atan2(y, x);

    return (toDegrees(θ) + 360) % 360;
  }

  function findNewPoint(loc, bearing, distance) {
    radius = 6371e3;
    var δ = Number(distance) / radius; // angular distance in radians
    var θ = toRadians(Number(bearing));

    var φ1 = toRadians(loc.lat);
    var λ1 = toRadians(loc.long);

    var sinφ1 = Math.sin(φ1),
      cosφ1 = Math.cos(φ1);
    var sinδ = Math.sin(δ),
      cosδ = Math.cos(δ);
    var sinθ = Math.sin(θ),
      cosθ = Math.cos(θ);

    var sinφ2 = sinφ1 * cosδ + cosφ1 * sinδ * cosθ;
    var φ2 = Math.asin(sinφ2);
    var y = sinθ * sinδ * cosφ1;
    var x = cosδ - sinφ1 * sinφ2;
    var λ2 = λ1 + Math.atan2(y, x);

    return { lat: toDegrees(φ2), long: (toDegrees(λ2) + 540) % 360 - 180 };
  }

  //  findLocation();

  return router;
})();
