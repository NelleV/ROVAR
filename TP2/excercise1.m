% Written by A. Vedaldi and A. Zisserman
%
% Modified by J. Sivic <josef.sivic@ens.fr>
% 10/10/2011
%


% Make sure to add path to vl_feat
setup;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stage A: data preparation and feature exraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: build visual vocabulary from a subset of 100 training images of each classes.
%
% The lists of all training images are given in ./data/image_lists/*_train.txt
%
% Hints:
%
% 1. Use function:
% vocabulary = computeVocabularyFromImageList(image_list);
%
% 2. To load a list of image names you can use function 
% list = textread('./data/image_lists/aeroplane_train.txt','%s')
%
% 3. Save the built vocabulary into ./data/vocabulary.mat using function:
% save('./data/vocabulary.mat','vocabulary');


% Write your code here:
%------------------------------------

% FIXME - needs to do that for all the different types of images

% FIXME check that this hasn't been done already, and skip this if it has.
list = textread('./data/image_lists/aeroplane_train.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/aeroplane_vocabulary.mat','vocabulary');

%------------------------------------



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: compute the spatial histogram representation of all training and
% test images
%
% To compute the spatial histogram representation use the following
% functions (e.g. for training images of the aeroplane class)
%
% names      = textread('./data/image_lists/aeroplane_train.txt','%s')
% histograms = computeHistogramsFromImageList(vocabulary, names);
% save(./data/histograms/aeroplane_train.mat,'histograms','names');
% 


% Write your code here:
%------------------------------------

% FIXME do this for all the files
% FIXME check that this hasn't been done already, and skip this if it has.
names = textread('./data/image_lists/aeroplane_train.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/aeroplane_train_hist.mat','histograms', 'names');


list = textread('./data/image_lists/background_train.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/background_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/background_train.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/background_train_hist.mat','histograms', 'names');


list = textread('./data/image_lists/background_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/background_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/background_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/background_val_hist.mat','histograms', 'names');


list = textread('./data/image_lists/aeroplane_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/aeroplane_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/aeroplane_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/aeroplane_val_hist.mat','histograms', 'names');


list = textread('./data/image_lists/car_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/acar_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/car_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/car_val_hist.mat','histograms', 'names');


list = textread('./data/image_lists/horse_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/horse_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/horse_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/horse_val_hist.mat','histograms', 'names');


list = textread('./data/image_lists/motorbike_train.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/motorbike_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/motorbike_train.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/motorbike_hist.mat','histograms', 'names');


list = textread('./data/image_lists/motorbike_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/motorbike_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/motorbike_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/motorbike_val_hist.mat','histograms', 'names');


list = textread('./data/image_lists/person_train.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/person_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/person_train.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/person_hist.mat','histograms', 'names');


list = textread('./data/image_lists/person_val.txt','%s');
vocabulary = computeVocabularyFromImageList(list);
save('./data/person_val_vocabulary.mat','vocabulary');

names = textread('./data/image_lists/person_val.txt','%s');
histograms = computeHistogramsFromImageList(vocabulary, names);
save('./data/histograms/person_val_hist.mat','histograms', 'names');

%------------------------------------




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: visualize computed features for one image
im                       = imread('./data/images/000060.jpg');
[keypoints, descriptors] = computeFeatures(im);
col     = keypoints(1,:);
row     = keypoints(2,:);
binsize = keypoints(4,:); % bin size sift descriptor (pixels).
                          
% recall that sift is composed of a spatial grid of 4x4 bins
radius  = binsize*2;      % visualize sift by a circle with radius of 2 bin widths.

figure(1); clf; imagesc(im);
vl_plotframe([col(1:50:end); row(1:50:end); radius(1:50:end)]); % visual keypoints as circles with radius scale
axis image;


% Let's determine the scales used, and the number of sift desc extracted
% for each scale
scale = 0;
scalecount = 0;
K = 36893;
for i=1:K,
    if keypoints(4, i) == scale
        scalecount = scalecount + 1;
    else
        disp(strcat('scale : ', num2str(scale)));
        disp(strcat('scale count : ', num2str(scalecount)));
        scale = keypoints(4, i);
        scalecount = 1;
    end;
end;
disp(strcat('scale : ', num2str(scale)));
disp(strcat('scale count : ', num2str(scalecount)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stage B: training a classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load training data
pos = load('./data/histograms/aeroplane_train_hist.mat') ;
neg = load('./data/histograms/background_train_hist.mat');
names = {pos.names{:}, neg.names{:}};
histograms = [pos.histograms, neg.histograms] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% load testing data
pos = load('data/histograms/aeroplane_val_hist.mat') ;
neg = load('data/histograms/background_val_hist.mat') ;
testNames = {pos.names{:}, neg.names{:}};
testHistograms = [pos.histograms, neg.histograms] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
    sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
    sum(testLabels > 0), sum(testLabels < 0)) ;

% l2 normalize the histograms before running the linear svm
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

% train the linear svm. the svm paramter c should be
% cross-validated. here for simplicity we pick a value that works
% well with all kernels.
c = 100 ;
[w, bias] = trainLinearSVM(histograms, labels, c) ;

% evaluate the scores on the training data
scores = w' * histograms + bias ;

% visualize the ranked list of images
figure(1) ; clf ; set(1,'name','ranked training images (subset)') ;
displayRankedImageList(names, scores)  ;

% visualize the precision-recall curve
figure(2) ; clf ; set(2,'name','precision-recall on train data') ;
vl_pr(labels, scores) ;

% visualize visual words by relevance on the first image
% displayRelevantVisualWords(names{1},w)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage C: Classify the test images and assess the performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Test the linar SVM
testScores = w' * testHistograms + bias ;

% Visualize the ranked list of images
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;

% Visualize the precision-recall curve
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_pr(testLabels, testScores) ;

% Print results
[drop,drop,info] = vl_pr(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage D: Learn the classifier for the other classes and assess its
% performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Write your code here:
%------------------------------------


%------------------------------------





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage E: Vary the image representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the following functions to remove spatial information from extracted
% histograms:
%
% histograms = removeSpatialInformation(histograms) ;
% testHistograms = removeSpatialInformation(testHistograms) ;


% Write your code here:
%------------------------------------



%------------------------------------





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage F: Vary the classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Take the square root of both the training and test histograms to be 
% used for the feature vectors. 
% Use the Matlab function "sqrt", which takes an elementwise 
% square root of a vector/matrix. 

% Write your code here:
%------------------------------------



%------------------------------------




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage G: Vary the number of training images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% You can use the following code to select only a subset of training data.
% Insert the code after you load the training and test data.
%
% fraction = .1 ; % fraction of data to keep
% fraction = .5 ;
fraction = +inf ;
sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
names = names(sel) ;
histograms = histograms(:,sel) ;
labels = labels(:,sel) ;
clear sel ;


% Write your code here:
%------------------------------------


%------------------------------------


