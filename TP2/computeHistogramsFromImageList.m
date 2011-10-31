function histograms = computeHistogramsFromImageList(vocabulary, names, cache)
% COMPUTEHISGORAMSFROMIMAGELIST  Compute historams for multiple images
%   HISTOGRAMS = COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES)
%   computes the histograms of visual words for the list of image
%   paths NAMES by applying iteratively
%   COMPUTEHISTOGRAMFROMIMAGE().
%
%   COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES, CACHE) caches
%   the results to the CACHE directory.

% Author: Andrea Vedaldi

if nargin > 2
  hasCache = true ;
else
  hasCache = false ;
  cache = [] ;
end
start = tic ;
histograms = cell(1,numel(names)) ;

% Note (J.Sivic): this loop can be executed in parallel by replacing below with 
% parfor i = 1:length(names)
% Paralelism was removed for clarity.
for i = 1:length(names) 
    if exist(names{i}, 'file')
    fullPath = names{i} ;
  else
    fullPath = fullfile('data','images',[names{i} '.jpg']) ;
  end
  if hasCache
    % try to retrieve from cache
    histograms{i} = getFromCache(fullPath, cache) ;
    if ~isempty(histograms{i}), continue ; end
  end
  fprintf('%d/%d Extracting histogram from %s (time remaining %.2fs)\n', i,length(names),fullPath, ...
          (length(names)-i) * toc(start)/i) ;
  histograms{i} = computeHistogramFromImage(vocabulary, fullPath) ;
  if hasCache
    % save to cache
    storeToCache(fullPath, cache, histograms{i}) ;
  end
end
histograms = [histograms{:}] ;

function histogram = getFromCache(fullPath, cache)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
if exist(cachePath, 'file')
  data = load(cachePath) ;
  histogram = data.histogram ;
else
  histogram = [] ;
end

function storeToCache(fullPath, cache, histogram)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
vl_xmkdir(cache) ;
data.histogram = histogram ;
save(cachePath, '-STRUCT', 'data') ;