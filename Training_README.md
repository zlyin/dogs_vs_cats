# datasets
	1. train set has 25000 images
	2. test set has 12500 images
	3. cats = 0, dogs = 1
	4. name style
		- cat.10003.jpg
		- dog.9997.jpg

================================================================
# building dataset hdf5 part

## config.py
	1. IMAGES_PATH = "./data/train/", total training images = 25000
	2. NUM_CLASSES = 2
	3. NUM_TRAINVAL_IMAGES = 1250 * NUM_CLASSES = 2500
	4. NUM_VAL_IMAGES = 1250 * NUM_CLASSES = 2500


## build_dogs_vs_cats.py
	1. use "config.py" file to define parameters; no need to feed in a bunch of
	   arguments
	2. follow dataset split strategy provided by Andre Ng
		for a fully labeled dataset, we can split:
		```
		- dataset = training data(60\%) + testing data(40\%)
		- testing data = validation(50\%) + true testing(50\%)
		- training data = train + trainval (a small portion, say 1/8)
		```
		noting that in this dataset, I treat trainset as the "original dataset", 
		since only the trainset is labeled
	4. split imagePaths & imageLabels into different parts & generate a HDF5 to
	   store such datasets
	5. while creating datasets, there 2 noting tips:
		- maintain the aspect ratio & resize the image to 256 * 256 (input_shape
			of AlexNet is (227, 227, 3))
		- record & write out "means of pixels of R/G/B channel" into a json file for 
		  "mean substration" use in **EVERY stage**
			- **mean substraction = data scaling, which scales data to "zero mean"**
			- another 0~1 scaling method  = "pixel / 255.0" 
	6. HDF5 files
		- raw images of dataset ~ 500M vs. train.hdf5 ~ 30G
		- exchange storage for avoiding I/O latency and futhermore improving 
		  training/predicting speed

================================================================
# training part

## Round 1 - use train AlexNet from the start
### train_alextnet.py
	1. aimming at accelerate training speed, load data from HDF5 file for
	   training \& tesing, which needs a HDF5DatasetGenerator class
	2. as for `trainGen`:
		- use 3 preprocessors:
			- `PatchPreprocessor(227, 227)` = randomly crop a 227 x 227 patch from
				original 256 x 256 image; 
			- `MeanPreprocessor(rMean, gMean, bMean)` = substract R/G/G mean
				values from each channel of image 
			- `ImageToArrayPreprocessor()` = convert image to Keras array
		- the reason why we extract patches for training is that we want the NN to 
		  learn local discrimitive patterns, not the whole semantics including the 
		  backgroud;
		- feed in a `ImageDataGenerator()` for data augmentation
	3. as for `valGen`
		- same as above except that no data augmentor is used; because in
			testing set evaluating, you don't apply data augmentation either!
	4. model compilation:
		- use `Adam(lr=1e-3)` as the optimizer
			- more accurate than `SGD`, but computing speed is slower
			- based on `RMSprop` algorithm;
	5. since we load data from HDF5DatasetGenerator, we use
	   `model.fit_generator` instead of `model.fit`
	   	- training_data = trainGen
		- validation_data = valGen
		- use `ModelCheckpoing` & `TrainingMonitor` as callbacks
	6. once training finish, **`HDF5DatasetGenerator.close()`**
	7. train 75 epochs


### eval_alexnet_crop_TTA.py
	1. use `testGen` to flow out data for fast prediction
		- use 3 preprocessors:
			- `SimplePreprocessor(227, 227)` = simply resize the original 256 x
				256 image to 227 x 227
				- **do not extract patch!**
			- `MeanPreprocessor(rMean, gMean, bMean)` 
			- `ImageToArrayPreprocessor()` 
	2. similarly, use `model.predcit_generator` instead of `model.predict`
	3. because the nature of binary classification, use **rank1_accuracy**
		- `rank5_accuracy` will simply return 100% accuracy, which is
			meaningless;
	4. **apply 10-crops oversampling/TTA** while evaluating
		- use `CropPreprocessor(227, 227)` to crop 227 x 227 images at 4 corners
			& the center of original image; apply`horizontal flip` if
			necessary; return 10 crops totally;
		- loop over each batch and each image in the batch, average predictions
			of 10 crops for each image;
		- still, use `rank1_accuracy`
		- **should expect accuracy improvement ~ 1 to 2%**
	5. accuracy:
		```	
		[INFO] predicting with AlexNet ...
		[INFO] evaluating on valset WITHOUT crop/TTA ...
		[INFO] rank-1 accuracy = 91.64%

		[INFO] evaluating on valset WITH crop/TTA ...
		[INFO] rank-1 accuracy = 91.92%

		[INFO] predicting with AlexNet2 (with padding)...
		[INFO] evaluating on valset WITHOUT crop/TTA ...
		[INFO] rank-1 accuracy = 90.60%

		[INFO] evaluating on valset WITH crop/TTA ...
		[INFO] rank-1 accuracy = 92.52%
		```

## Round2 - use ResNet50 as feature extractor + logistic regressor as head
### extract_features_resnet50.py
	1. feature extraction using `ResNet50(weights=imagenet)` for transfer learning
		- strategy = FE by NN & ML classifier to predict
			- ImageNet definitely contains "dogs" & "cats" category => dataset
				similar
			- this dataset is very small
		- extract features from **training set**
	2. list all imagePaths in the `./data/train` and split into batches for
	   prediction
	   	- **need to `random.shuffle(imagePaths)`**, otherwise, one batch would
			hold data with same label!
		- get labels from imagePath
	3. use ResNet50 as feature extractor:
		- noting that `ResNet50(weights=imagenet, include_top=False)` will
			acutally drop the last `Dense(2048)` layer and the penultimate
			`GlobalAveragePooling2D` layer; therefore, the output volumn from
			such backbone has dimension of (batch, 7, 7, 2048);
		- logistic regressor only takes in vector for each sample
			- need to **reshape/flatten the features into (batch, num_of_columns)**
		- 2 versions:
			- perform model surgery to build a new model 
				- `basemodel = ResNet50()`
				- add a `GlobalAveragePooling2D` layer
				- output = (batch, 2048)
			- accumulative mulplication to get `num_of_columns = 7 x 7 x 2048`
	4. create `HDF5DatasetWirter` to convert data batches into HDF5 file	
		- use keras utilis tools to preprocess images:
			- `keras.preocessing.image.load_img` to load in 227 x 227 image
			- `keras.preprocessing.image.img_to_array` to covert image to keras
				array
			- substract pixel mean of RGB by
				`keras.applications.imagenet_utils.preprocess_input`
		- convert a list of images/arrays to an array
			- `np.vstack([image array])`
		- use `model.precit()` to create features
			- reshape into (batch, num_of_columns)
	5. **`HDF5DatasetWriter.close()`**


### train_logistic_regressor.py
	1. train a **logistic regressor** as the classifier on top of extracted features
	2. split the features.hdf5 file (generated from `./data/train`) into 2 parts
	   by setting a threshold index, `train : val = 4 : 1`;
	3. use `model = GridSearchCV(cls, params, scoring, cv, n_jobs, verbose,
	   refit)` 
	   to fine tune a logistic regressor
	   	- params = `{"C" : [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1]}` 
	   	- model.fit(traindata, trainlabel)
	   	- retrieve the **best params** by `model.best_params_`
	   	- seralize the **best estimator** by `f.write(pickle.dumps(model.best_estimator_))`
	4. featrues extractor = ResNet50 backbone **WITH** GlobalAveragePooling2D
		- test 1:
	   		- GridSearchCV = `LogisticRegression(solver="lbfgs")` + cv = 5 + `["accuracy", "neg_log_loss"]`
	   		```
			[INFO] best hyperparameters are {'C': 0.01}
			[INFO] best score is -0.03091793724858644
			[INFO] evaluating on val set...
						  precision    recall  f1-score   support

					 cat       0.99      0.99      0.99      2483
					 dog       0.99      0.99      0.99      2517

				accuracy                           0.99      5000
			   macro avg       0.99      0.99      0.99      5000
			weighted avg       0.99      0.99      0.99      5000
			[INFO] accuracy score : 0.9908
			```
			- public LB score = 
	   			- without TTA:
					- without clip - 0.07147
					- with `np.clip(probs, 0.02, 0.98)` - 0.06578
				- with TTA
					- without clip - 0.06769
					- with `np.clip(probs, 0.02, 0.98)` - **0.06297/Rank=134, best score so far!**
		- test 2:
			- set *cv=10*, scores are exactly same test 1; 
		- test 3:
			- set *scoring=["neg_log_loss"]*, scores are exactly same as test 1
		- test 4:
			- add *build a Pipeline(steps=[("pca", pca), ("logreg", logreg)])* 
			```
			[INFO] best score is -0.03346863651597691
			[INFO] best hyperparameters are {'logreg__C': 0.023357214690901212, 'pca__n_components': 60}
			[INFO] accuracy score : 0.9888
			```
			- public LB score:
				- with TTA
					- without clip - 0.07194
					- with `np.clip(probs, 0.02, 0.98)` - 0.0661 
		- test 5:
			- adjust `logreg__C` & `pca__n_components` candidates
			```
			[INFO] best score is -0.032569273157581
			[INFO] best hyperparameters are {'logreg__C': 0.012742749857031334, 'pca__n_components': 120}
			[INFO] accuracy score : 0.9894
			```
			- public LB score:
				- with TTA
					- without clip - 0.07035
					- with `np.clip(probs, 0.02, 0.98)` - 0.06505

### predict_logistic_regressor.py
	1. use pretrained models to predict on `./data/test` & generate submission.csv
	2. featrues extractor = ResNet50 backbone *WITH* GlobalAveragePooling2D
	3. `parser.add_argument("-tta", "--TTA", default=True, type=lambda x : str(x).lower()=="true")`
		- note the usage of `lambda x` to feed a **boolean**
	4. **loaded imagePaths are not stored in the order of submission.csv!**
		- have to loop over each imageId and `sub.loc[sub["id"] == idx, "label"] = predictions[i]`
	5. **np.clip(probs, 0.02, 0.98) is helpful!**

## Round 3 - use ResNet50 as backbone, finetune its head
	1. recreate the HDF5 dataset
		- e.g the orginal val dataset ONLY has dogs!
	2. use a new NN
		- backbone = ResNet50 (without GloAvgPool2D layer)
		- head = *BN => GloAvgPool2D => Dense(2) => Activation("softmax")*
	2. do several test towards the **preprocessors combo**
		- hyperparameters
			```
			BATCH_SIZE = 128
			EPOCHS = 5
			Adam(lr=1e-3)
			```	
		- test 1
			- trainGen = [Patch, MeanSubstrct, ImgToArray]
			- trainvalGen = [Patch, MeanSubstrct, ImgToArray]
			- aug for trainGen:	
				```
				aug = ImageDataGenerator(
					rotation_range=20,
					zoom_range=0.15,
					width_shift_range=0.2,
					height_shift_range=0.2,
					shear_range=0.15,
					horizontal_flip=True,
					fill_mode="nearest",
					)
				```
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.03867444735292333
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       1.00      0.97      0.98      1280
						   1       0.97      1.00      0.98      1220

					accuracy                           0.98      2500
				   macro avg       0.98      0.98      0.98      2500
				weighted avg       0.98      0.98      0.98      2500
				[INFO] rank-1 accuracy = 98.40%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] log loss = 0.038421075657996046
				[INFO] rank-1 accuracy = 98.64%
				```
			- on public LB
				- without TTA - 0.10407
				- with TTA - 0.10148
		- test 2
			- trainGen = *[Patch, MeanSubstrct, ImgToArray]*
			- trainvalGen = *[Patch, MeanSubstrct, ImgToArray]*
			- aug for trainGen:	
				```
				aug = ImageDataGenerator(
						rotation_range=20,
						zoom_range=0.05,
						width_shift_range=0.05,
						height_shift_range=0.05,
						shear_range=0.05,
						horizontal_flip=True,
						fill_mode="nearest",
						)
				```
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.03856617170537303
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       1.00      0.98      0.99      1275
						   1       0.98      1.00      0.99      1225

					accuracy                           0.99      2500
				   macro avg       0.99      0.99      0.99      2500
				weighted avg       0.99      0.99      0.99      2500

				[INFO] rank-1 accuracy = 98.44%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] rank-1 accuracy = 98.52%
				```
			- on public LB
				- **without TTA - 0.10049**
				- **with TTA - 0.09784**
		- test 3
			- **remove MeanSubstraction** when compared to test2
				- trainGen = *[Patch, ImgToArray]*
				- trainvalGen = *[Patch, ImgToArray]*
			- aug is same as that of test 2
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.05279085544339009
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       0.99      0.97      0.98      1281
						   1       0.97      0.99      0.98      1219

					accuracy                           0.98      2500
				   macro avg       0.98      0.98      0.98      2500
				weighted avg       0.98      0.98      0.98      2500

				[INFO] rank-1 accuracy = 97.80%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] log loss = 0.04945724408615818
				[INFO] rank-1 accuracy = 97.92%
				```
			- on public LB
				- without TTA - 0.11048
				- with TTA - 0.10701
				- **MeanSubstraction is helpful!**
		- test 4
			- **replace Patch with Simple preprocessor** when compared to test3
				- trainGen = *[Simple, ImgToArray]*
				- trainvalGen = *[Simple, ImgToArray]*
			- aug is same as that of test 2
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.06262893622020783
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       0.99      0.97      0.98      1283
						   1       0.97      0.99      0.98      1217

					accuracy                           0.98      2500
				   macro avg       0.98      0.98      0.98      2500
				weighted avg       0.98      0.98      0.98      2500
				[INFO] rank-1 accuracy = 97.88%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] log loss = 0.05055090900043961
				[INFO] rank-1 accuracy = 98.08%
				```
			- on public LB
				- without TTA - 0.10434 
				- with TTA - 0.10139
				- **SimplePreprocessor/resize is better than PatchPreprocessor/random crop w/o MeanSubstrct**
		- test 5
			- use a new preprocessor combo
				- trainGen = *[Simple, MeanSubstract, ImgToArray]*
				- trainvalGen = *[Simple, MeanSubstract, ImgToArray]*
			- aug is same as that of test 2
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.047123435600522234
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       0.99      0.98      0.98      1272
						   1       0.98      0.99      0.98      1228

					accuracy                           0.98      2500
				   macro avg       0.98      0.98      0.98      2500
				weighted avg       0.98      0.98      0.98      2500
				[INFO] rank-1 accuracy = 98.40%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] log loss = 0.037465767340646765
				[INFO] rank-1 accuracy = 98.44%
				```
			- on public LB
				- with routine *[Aspect, ImgToArr]* preprocessors
					- without TTA - 0.10301 
					- with TTA - 0.10001
					- **add MeanSubstract preprocessor during Train is helpful;**
						**but Patch + MeanSubstract + ImgToArr is better!**
				- with new *[Simple, MeanSubstract, ImgToArray]* preprocessors for test
					- without TTA - 0.16560 
					- with TTA - 0.15600 
					- *[Simple, MeanSubstract, ImgToArray] test is harmful!*
		- test 6
			- **replace Simple with AspectAware** based on test 5
				- trainGen = *[AspectAware, MeanSubstract, ImgToArray]*
				- trainvalGen = *[AspectAware, MeanSubstract, ImgToArray]*
			- aug is same as that of test 2
			- on val set performance
				```
				[INFO] evaluating on valset WITHOUT crop/TTA ...
				[INFO] log loss = 0.04952111005278631
				[INFO] classification report 
							   precision    recall  f1-score   support

						   0       0.99      0.97      0.98      1275
						   1       0.97      0.99      0.98      1225

					accuracy                           0.98      2500
				   macro avg       0.98      0.98      0.98      2500
				weighted avg       0.98      0.98      0.98      2500
				[INFO] rank-1 accuracy = 98.44%

				[INFO] evaluating on valset WITH crop/TTA ...
				[INFO] log loss = 0.039185087170255886
				[INFO] rank-1 accuracy = 98.40%
				```
			- on public LB
				- with routine *[Aspect, ImgToArr]* preprocessors
					- **without TTA - 0.10057**
					- **with TTA - 0.09783**
					- **[Aspect, MeanSubstract, ImgToArray] is highly similar to [Patch, MeanSubstract, ImgToArray] of test 2**
				- with new *[Simple, ImgToArray]* preprocessors
					- without TTA - 0.10765 
					- with TTA - 0.10654
					- **not as good as routine combo for prediction**
				- with new *[Aspect, MeanSubstract, ImgToArray]* preprocessors for test
					- without TTA - 0.15805
					- with TTA - 0.14919 
					- *[Aspect, MeanSubstract, ImgToArray]/add MeanSubstract in tset is harmful*
		- preprocessors conclusion:
			- for **trainGen & trainvalGen**:
				- [AspectAware/Patch, MeanSubstract, ImgtoArray] are the best
					- add MeanSubstract is necessary;
			- for **test set**: 
				- [Aspect, ImgToArray] are the best
		- try clipping:
			- test6 model w/o TTA:
				- round probs to `%.4f` - 0.13632
				- round probs to `%.4f` and `np.clip(probs, 0.02, 0.98)` - **0.09379** 
				- only `np.clip(probs, 0.02, 0.98)` - **0.09379**
				- only `np.clip(probs, 0.05, 0.95)` - 0.11524
			- test6 model with TTA:
				- only `np.clip(probs, 0.02, 0.98)` - **0.09199**
				- only `np.clip(probs, 0.05, 0.95)` - 0.11359

================================================================
# submission part

## predict_alexnet.py
	1. use pretrained models to predict on `./data/test` & generate
	   submission.csv
	2. AlexNet results (w.r.t. logloss):
		- AlexNet(without padding) - public LB score = 1.81479
		- **AlexNet(without padding) + TTA - public LB score = 1.66289** ??
		- AlexNet2(with padding) - public LB score = 2.19301
		- AlexNet2(with padding) + TTA - public LB score = 1.97178



