const PARAMIDX = Object.freeze({
  'N': 0,
  'NetSR': 1,
  'NetinpScaling': 2,
  'BiasScaling': 3,
  'TychonovAlpha': 4,
  'washoutLength': 5,
  'learnLength': 6,
  'TychonovAlphaReadout': 7,
  'LR': 8,
  'windowSize': 9
});

const ENVIDX = Object.freeze({
  'MORPH': 0,
  'STRETCH': 1,
  'LR': 2,
  'SR': 3,
  'GAIN': 4
});

const MODELTYPE = Object.freeze({
  'CONSTWINDOW': 0,
  'VARIABLEWINDOW': 1,
});

function loadModels(onLoaded, onProgress) {
  function loadFloatBin(url, onLoad) {
    console.log("Loading " + url);
    var req = new Request(url);

    window.caches.open('conceptular').then(function(cache) {
      return cache.match(req).then(function(response) {
        console.log(response);
        return response || fetch(req).then(function(response) {
          console.log("adding to cache from network");
          cache.put(req, response.clone());
          return response;
        })
        // .then(function(data) {
        //   console.log("fetched from network");
        //   console.log(data);
        // });
      }).then(function(response) {
        console.log("fetched");
        console.log(response);
        response.arrayBuffer().then(data => {
          console.log(data);
          onLoad(data);
        })
      });
    })


  }

  // const modelRoot = "models/snare_bongo_";
  // const modelRoot = "models/hihats_";
  // const modelRoot = "models/clavetri_";
  let modelRootList = [
    ["models/hihats_", MODELTYPE.CONSTWINDOW],
    ["models/clavetri_", MODELTYPE.CONSTWINDOW],
    ["models/snare_bongo_", MODELTYPE.CONSTWINDOW],
    ["models/kicks_", MODELTYPE.VARIABLEWINDOW],
  ];
  console.log(modelRootList);
  // const modelRoot = "models/kicks_";
  models = []
  // model = [];
  // model.modelType = MODELTYPE.VARIABLEWINDOW;
  // model.modelType = MODELTYPE.CONSTWINDOW;
  let progress = 0;
  let progressPerModel = 1.0 / modelRootList.length;
  let progressIncrement = 0.0;
  function addModel(modelRoots) {
    console.log("Loading " + modelRoots[0]);
    model = {};
    model.modelType = modelRoots[0][1];
    let filenames = ['Wbias', 'Cs', 'Wout', 'W', 'params'];
    if (model.modelType == MODELTYPE.VARIABLEWINDOW) {
      filenames = filenames.concat(['index1', 'index2']);
    }
    models[models.length] = model;
    progressIncrement = progressPerModel / filenames.length;
    loadBinFiles(modelRoots, filenames, model);
  }

  function loadBinFiles(modelRoots, listOfFiles, container) {
    loadFloatBin(modelRoots[0][0] + listOfFiles[0] + ".bin", function(e) {
      // container[listOfFiles[0]] = new Float32Array(this.response);
      // console.log(e);
      container[listOfFiles[0]] = new Float32Array(e);
      console.log("Completed loading: " + listOfFiles[0])
      progress += progressIncrement;
      onProgress(progress);
      if (listOfFiles.length > 1) {
        loadBinFiles(modelRoots, listOfFiles.slice(1), container);
      } else {
        if (container.modelType == MODELTYPE.VARIABLEWINDOW) {
          //do some setup calculations
          container.sound1End = container.index1[container.index1.length - 1]
          container.sound2End = container.index2[container.index2.length - 1]
          container.longest = Math.max(container.sound1End, container.sound2End);
          container.sound2CStartIdx = container.index1.length - 1;
        }
        if (modelRoots.length > 1) {
          addModel(modelRoots.slice(1));
        } else {
          onLoaded();
        }
      }
    });
  }
  addModel(modelRootList);



}



function render(modelIdx, pattern, envelopes, stepLengthMs, renderHandler, stepHandler) {
  var w;
  console.log("rendering model " + modelIdx);
  // const time = await tf.time(async () => {
  const startTime = Date.now();

  let model = models[modelIdx];

  const N = model['params'][PARAMIDX.N];
  const numConceptors = model['Cs'].length / (N * N);
  const conceptorBlockBStart = numConceptors / 2;
  let modelRenderLength;

  if (model.modelType == MODELTYPE.VARIABLEWINDOW) {
    modelRenderLength = model.longest;
  } else {
    modelRenderLength = numConceptors * model['params'][PARAMIDX.windowSize] / 2.0;
  }
  const stepLength = Math.floor(stepLengthMs / 1000.0 * 22050);
  console.log("StepLen: " + stepLength);
  const sequenceRenderLength = stepLength * pattern[modelIdx].length;
  let sequencerPos = 0;
  let modelPos = -1;
  let patternIdx = modelIdx;
  let modelSpeed = 1;

  console.log("Rendering " + sequenceRenderLength + " samples...");
  let W = tf.tensor2d(model['W'], [N, N]);
  let Wbias = tf.tensor1d(model['Wbias']);
  let Wout = tf.tensor1d(model['Wout']);
  let Cs = tf.tensor3d(model['Cs'], [numConceptors, N, N]);
  let x = tf.randomNormal([N], 0, 0.5);
  // let xOld = tf.zerosLike(x);
  // let wTarget = tf.zerosLike(x);
  // let zA = tf.zerosLike(x);
  // let zB = tf.zerosLike(x);
  // let w = tf.tensor1d([]);
  w = new Float32Array(sequenceRenderLength).fill(0);
  let lr = tf.scalar(model['params'][PARAMIDX.LR])
  let lrOneMinus = tf.scalar(1.0 - model['params'][PARAMIDX.LR])
  let tensorOne = tf.tensor1d([1]);
  let C;
  tf.tidy(() => {
    C = Cs.slice(0, 1).as2D(N, N);
    tf.keep(C);
  });
  //washout
  for (let i = 0; i < model['params'][PARAMIDX.washoutLength]; i++) {

    tf.tidy(() => {
      let xOld = tf.clone(x);
      let wTarget = tf.dot(W, x);
      let zA = tf.mul(lrOneMinus, xOld);
      let zB = tf.mul(lr, tf.tanh(tf.add(wTarget, Wbias)));
      let z = tf.add(zA, zB);
      x.dispose();
      x = tf.dot(C, z);
      tf.keep(x);
    });
  }
  C.dispose();

  // rendering
  let sound1CIdx = 0;
  let sound2CIdx = 0;
  let sound2PreIndex = 0;

  var writeCount = 0;
  var batchSize = 350;
  let step = -1;
  let nextStep = 0;

  function cleanup() {

    // tf.tensor1d(w).data().then(renderHandler);
    renderHandler(w);

    //cleanup
    W.dispose();
    C.dispose();
    Wbias.dispose();
    Wout.dispose();
    Cs.dispose();
    x.dispose();
    // w.dispose();
    lr.dispose();
    lrOneMinus.dispose();
    // xOld.dispose();
    tensorOne.dispose();
    // wTarget.dispose();
    // zA.dispose();
    // zB.dispose();
    console.log("cleanup");
    // console.log(w);
    console.log("Time: " + (Date.now() - startTime) + " ms");
  }

  function writeToSequence(v) {
    w[this.i] = v[0] * envelopes[ENVIDX.GAIN].scan(this.normPos);
    this.sample.dispose();
    writeCount--;
    if (0 == writeCount) {
      if (sequencerPos == sequenceRenderLength) {
        cleanup();
      } else {
        renderBatch();
      }
    }
  };

  function renderBatch() {
    let batchEnd = sequencerPos + batchSize;
    while (sequencerPos < Math.min(batchEnd, sequenceRenderLength)) {
      if (sequencerPos == nextStep) {
        step++;
        let adjustedStepLength = stepLength;
        //swing test - implement later
        // if (step % 2 == 0) {
        //   adjustedStepLength *= 1.5;
        // } else {
        //   adjustedStepLength *= 0.5;
        // }
        nextStep += Math.floor(adjustedStepLength);
        // let step = sequencerPos / stepLength;
        stepHandler(step + 1);
        if (pattern[patternIdx][step]) {
          modelPos = 0;
          sound1CIdx = 0;
          sound2CIdx = 0;
          sound2sound2PreIndex = 0;
        }
      }
      // let wave = tf.zeros([1]);
      if (modelPos >= 0) {
        if (model.modelType == MODELTYPE.VARIABLEWINDOW) {
          let modelPosFloor = Math.floor(modelPos);
          if ((sound1CIdx < model.index1.length - 2) && (modelPosFloor == model.index1[sound1CIdx + 1])) {
            sound1CIdx++;
            console.log("s1: " + sound1CIdx)
          }
          if ((sound2sound2PreIndex < model.index2.length - 2) && (modelPosFloor == model.index2[sound2sound2PreIndex + 1])) {
            sound2sound2PreIndex++
            sound2CIdx = sound2sound2PreIndex + model.sound2CStartIdx;
            console.log("s2: " + sound2sound2PreIndex)

          }
        } else {
          sound1CIdx = Math.floor(modelPos / model['params'][PARAMIDX.windowSize]);
          sound2CIdx = sound1CIdx + conceptorBlockBStart;
        }
        tf.tidy(() => {
          let normalisedPosition = sequencerPos / sequenceRenderLength;
          currLR = model['params'][PARAMIDX.LR] * (envelopes[ENVIDX.LR].scan(normalisedPosition) * 2.0);
          lr = tf.scalar(currLR);
          lrOneMinus = tf.scalar(1.0 - currLR);

          let morph = envelopes[ENVIDX.MORPH].scan(normalisedPosition);
          C.dispose();
          let C1 = tf.mul(Cs.slice(sound1CIdx, 1).as2D(N, N), tf.scalar(morph));
          let C2 = tf.mul(Cs.slice(sound2CIdx, 1).as2D(N, N), tf.scalar(1.0 - morph));
          let CCombo = tf.add(C1, C2);
          C = tf.mul(CCombo, ((envelopes[ENVIDX.SR].scan(normalisedPosition) * 0.4) + 0.8));
          let xOld = tf.clone(x);
          let wTarget = tf.dot(W, x);
          let zA = tf.mul(lrOneMinus, xOld);
          let zB = tf.mul(lr, tf.tanh(tf.add(wTarget, Wbias)));
          let z = tf.add(zA, zB);
          x.dispose();
          x = tf.dot(C, z);
          let newSample = Wout.dot(x.concat(tensorOne)).as1D();
          writeCount++;
          // w[sequencerPos] = newSample.dataSync()[0];
          let functionCall = writeToSequence.bind({
            i: sequencerPos,
            normPos: normalisedPosition,
            sample: newSample
          })
          newSample.data().then((v) => {
            functionCall(v);
          })
          // console.log(newSample);
          tf.keep(x);
          tf.keep(C);
          tf.keep(newSample);
        });

        modelSpeed = envelopes[ENVIDX.STRETCH].scan(sequencerPos / sequenceRenderLength) * 2.0;
        modelPos += modelSpeed;
        if (sequencerPos % 100 == 0) {
          console.log([sequencerPos, modelSpeed, modelPos]);
        }
        // console.log(modelPos);
        if (modelPos >= modelRenderLength) modelPos = -1;
      } else {
        w[sequencerPos] = 0;
      }

      sequencerPos++;
    }
    if (writeCount == 0) {
      if (sequencerPos == sequenceRenderLength) {
        cleanup();
      } else {
        setTimeout(
          renderBatch, 1);
      }

    }
  }
  console.log("rendering batch");
  renderBatch();


  // });
  // console.log(time);
}
