import cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM
# from featureextraction import extract_features
from featureextraction import extract_features
import warnings

warnings.filterwarnings("ignore")

source = "trainingData/"

dest = "Speakers_models/"
train_file = "trainingDataPath.txt"
file_paths = open(train_file, 'r')

count = 1
# Extracting features for each speaker (8 files per speakers)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print path

    # read the audio
    sr, audio = read(source + path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 8 files of speaker are concatenated, then do model training
    # -> if count == 8: --> edited below
    if count == 8:
        gmm = GMM(n_components=10, n_iter=200, covariance_type='full', n_init=3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("-")[0] + ".gmm"
        cPickle.dump(gmm, open(dest + picklefile, 'w'))
        print '+ modeling completed for speaker:', picklefile, " with data point = ", features.shape
        features = np.asarray(())
        count = 0
    count = count + 1
