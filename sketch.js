let poseNet;
let video;
let predictions = [];

let model;
let targetLabel = [];
let state = 'collection';

let nnResults;
let loopBroken = false;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "predictions"
  // with an array every time new predictions are made
  // Listen to new 'pose' events
  poseNet.on('pose', function (results) {
    predictions = results;
  });

  let options = {
    inputs: 34,
    outputs: 10,
    task: 'regression',
    debug: 'false',
  };

  model = ml5.neuralNetwork(options);
  // autoStartPredict();

  // Hide the video element, and just show the canvas
  video.hide();

  createButton('Load Model').mousePressed(onLoadModelClick);
  createButton('Start Prediction').mousePressed(onPredictClick);
}

function autoStartPredict() {
  if (state == 'prediction') {
    onLoadModelClick();
    onPredictClick();
  }
}

function dataLoaded() {
  console.log(model.data);
}

function modelReady() {
  console.log('Model ready!');
}

function draw() {
  image(video, 0, 0, width, height);
  if (predictions.length > 0) {
    drawSkeleton(predictions);
    drawKeypoints(predictions);
  }
  restartPredictions();
}

// The following comes from https://ml5js.org/docs/posenet-webcam // A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < predictions.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = predictions[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255);
        stroke(20);
        strokeWeight(4);
        ellipse(round(keypoint.position.x), round(keypoint.position.y), 8, 8);
      }
    }
  }
}
// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < predictions.length; i++) {
    let skeleton = predictions[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255);
      strokeWeight(1);
      line(
        partA.position.x,
        partA.position.y,
        partB.position.x,
        partB.position.y
      );
    }
  }
}

function mousePressed() {
  if (predictions[0] != undefined) {
    let inputs = getInputs();
    console.log(getInputs());
    if (state == 'collection') {
      let target = targetLabel;
      if (targetLabel != undefined) {
        model.addData(inputs, target);
        console.log(`Data recorded for label ${targetLabel}`);
      } else {
        console.log('Target label not set.');
      }
    } else if (state == 'prediction') {
      model.predict(inputs, gotResults);
    }
  }
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  console.log(results);
  nnResults = results;
  classify();
}

function keyPressed() {
  if (key == 't') {
    console.log('starting training');
    state = 'training';
    model.normalizeData();
    let options = {
      epochs: 50,
    };
    model.train(options, whileTraining, finishedTraining);
  } else if (key == 's') {
    model.saveData();
  } else if (key == 'm') {
    model.save();
  } else if (key == 'r') {
    targetLabel = [
      random(),
      random(),
      random(),
      random(),
      random(),
      random(),
      random(),
      random(),
      random(),
      random(),
    ];
    console.log(targetLabel);
  }
}

function whileTraining(epoch, loss) {
  console.log(epoch, loss);
}
function finishedTraining() {
  console.log('finished training');
}

function classify() {
  if (predictions[0] != undefined) {
    let inputs = getInputs();
    model.predict(inputs, gotResults);
  } else {
    loopBroken = true;
  }
}

function onPredictClick() {
  state = 'prediction';
}
function onLoadModelClick() {
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  model.load(modelInfo, () => console.log('Model Loaded.'));
}

function restartPredictions() {
  if (loopBroken) {
    loopBroken = false;
    classify();
  }
}

function getInputs() {
  let keypoints = predictions[0].pose.keypoints;
  let inputs = [];
  for (let i = 0; i < keypoints.length; i++) {
    inputs.push(keypoints[i].position.x);
    inputs.push(keypoints[i].position.y);
  }
  return inputs;
}
