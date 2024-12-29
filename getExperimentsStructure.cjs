const fs = require('fs');
const path = require('path');

function getModels(dir) {
  return fs.existsSync(dir)
    ? fs.readdirSync(dir).map((modelName) => {
        const metaDir = path.join(dir, modelName, 'meta');
        const resultsDir = path.join(dir, modelName, 'generations');

        if (!fs.existsSync(dir)) return [];

        const modelPath = path.join(dir, modelName);

        // Initialize model object
        const model = {
          name: modelName,
          type: 'model',
          path: modelPath,
          experiments: []
        };
        if (fs.existsSync(metaDir)) {
          const metaFiles = fs
            .readdirSync(metaDir)
            .filter((file) => path.extname(file) === '.json');
          model.experiments = metaFiles.map((metaFile) => {
            const resultsBaseName = path.basename(
              metaFile.replace(/_dp-\d+\.\d+/g, ''),
              '.json'
            );
            const metaFilePath = path.join(metaDir, metaFile);

            const resultsFilePath = path.relative(
              __dirname,
              path.join(resultsDir, resultsBaseName + '.jsonl')
            );

            return {
              name: metaFile,
              type: 'meta',
              meta: JSON.parse(fs.readFileSync(metaFilePath, 'utf8')),
              path: metaFilePath,
              results: fs.existsSync(resultsFilePath)
                ? {
                    name: resultsBaseName,
                    type: 'results',
                    path: resultsFilePath
                  }
                : null // No corresponding results file
            };
          });
        }

        return model;
      })
    : [];
}

function getTopKs(dir) {
  // Retrieve the list of top_k directories, each containing models
  return fs.existsSync(dir)
    ? fs.readdirSync(dir).map((topK) => {
        const topKPath = path.join(dir, topK);
        const models = getModels(topKPath);

        return {
          name: topK,
          type: 'top_k',
          path: topKPath,
          models
        };
      })
    : [];
}

function getSplits(dir) {
  return fs.existsSync(dir)
    ? fs.readdirSync(dir).map((split) => {
        const splitPath = path.join(dir, split);
        const setups = getSetups(splitPath);

        return {
          name: split,
          type: 'split',
          path: splitPath,
          setups
        };
      })
    : [];
}

function getSetups(dir) {
  return fs.readdirSync(dir).reduce((acc, setup) => {
    const setupPath = path.join(dir, setup);
    const isDenseSetup = setup.toLowerCase().includes('dense');

    if (isDenseSetup) {
      const denseSplitPath = path.join(setupPath, fs.readdirSync(setupPath)[0]);
      const allDenseSetups = fs.readdirSync(setupPath).map((denseSetup) => {
        const denseSetupName = `${setup}/${denseSetup}`;
        const denseSetupPath = path.join(setupPath, denseSetup);
        const topKs = getTopKs(denseSetupPath);
        return {
          name: denseSetupName,
          type: 'top_k',
          dense: true,
          topKs
        };
      });
      topKs = getTopKs(denseSplitPath);
      return [...acc, ...allDenseSetups];
    } else {
      // Regular setups have splits directly under the setup
      topKs = getTopKs(setupPath);
      return [
        ...acc,
        {
          name: setup,
          type: 'top_k',
          path: setupPath,
          dense: false,
          topKs
        }
      ];
    }
  }, []);
}

function parseDataDir(rootDir) {
  // Retrieve the list of datasets, each containing setups
  return fs.readdirSync(rootDir).map((dataset) => {
    const datasetPath = path.join(rootDir, dataset);
    const splits = getSplits(datasetPath);

    return {
      name: dataset,
      type: 'dataset',
      path: datasetPath,
      splits
    };
  });
}

const dataDir = path.join(__dirname, 'basement');
const datasets = parseDataDir(dataDir);

fs.writeFileSync('./experiments.json', JSON.stringify(datasets, null, 2));
console.log('Experiment structure saved to experiments.json');
