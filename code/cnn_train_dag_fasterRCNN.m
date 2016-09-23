function [net,stats] = cnn_train_dag_fasterRCNN(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Todo: save momentum with checkpointing (a waste?)

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
%liu@0812 added:
opts.batchPerImage = 256;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
%0921 added
opts.use_conv4 = false;

%opts.derOutputs = {'objective', 1} ;
%0612 changed to have multiple objectives
%opts.derOutputs = {'objective', 1, 'objective2', 0.001} ;
% liu@0812: same weights for both cls and reg
opts.derOutputs = {'loss_cls', 1, 'loss_bbox', 10} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

%0921 added
%0921_conv4: 'loss_cls', 1, 'loss_bbox', 40
%0922_conv4: 'loss_cls', 1, 'loss_bbox', 20
if opts.use_conv4
    opts.derOutputs = {'loss_cls', 1, 'loss_bbox', 10} ;
end

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, stats] = loadState(modelPath(start)) ;
end
%0528 added
% lrcnt = 1;
% lrincreflag = 0;
for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  %0528 changed
%   if lrcnt > numel(opts.learningRate)
%       fprintf('Training stopped at epoch %d due to saturated learning rate\n', epoch) ;
%       break;
%   end
  %state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  %state.learningRate = opts.learningRate(lrcnt) ;
  %state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  %liu@0913: change learning rate according to #epoch
  state.learningRate = opts.learningRate(epoch) ;
  state.train = opts.train; % 0818: no shuffle for debugging
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
    clear net_ stats_ stats__ savedNet_ ;
  end

  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  %0528 added: if the validation error stops decreasing, lower learning
  %rate by an order of magnitude
  %clf
%   clsErrArr = [stats.val.loss_cls];
%   bboxErrArr = [stats.val.loss_bbox];
%   %reg
%   %valerrArr = [stats.val.objective];
%   if length(clsErrArr) > 1 && ( clsErrArr(end) >= clsErrArr(end-1) || bboxErrArr(end) >= bboxErrArr(end-1) )
%       lrincreflag = lrincreflag + 1;
%       if lrincreflag == 1
%           lrcnt = lrcnt + 1;
%           lrincreflag = 0;
%       end
% %   else
% %       lrincreflag = max(lrincreflag - 1, 0);
%   end
  
  if opts.plotStatistics
    figure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t = 1:opts.batchSize:numel(subset)
    %if t == 279
    %   fprintf('Problem begins next time\n'); 
    %end
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    % for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
%     batchStart = t + (labindex-1) + (s-1) * numlabs ;
%     batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
%     batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    batch = subset(t);
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    %inputs = state.getBatch(state.imdb, batch) ;
    %liu@0809: only give a single image as args to reduce stack overload
    inputs = state.getBatch(state.imdb.data(batch)) ;
    
    %liu@0810: currently prefetch is disabled
    if opts.prefetch
%       if s == opts.numSubBatches
%         batchStart = t + (labindex-1) + opts.batchSize ;
%         batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
%       else
%         batchStart = batchStart + numlabs ;
%       end
%       nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        if t ~= numel(subset)
            nextBatch = subset(t+1);
            state.getBatch(state.imdb.data(nextBatch));
        end
    end

    if strcmp(mode, 'train')
        net.mode = 'normal' ;
        %liu@0810: masked
        %net.accumulateParamDers = (s ~= 1) ;
        net.eval(inputs, opts.derOutputs) ;
    else
        net.mode = 'test' ;
        net.eval(inputs) ;
    end
  %end

  % =========liu@0908: get the result for comparison===============
  rst = check_error(net.vars, opts.use_conv4);
  %disp(rst);
  % =============================================
  
  % extract learning stats
  stats = opts.extractStatsFn(net) ;

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

%   fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
%     mode, ...
%     state.epoch, ...
%     fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
%     stats.num/stats.time * max(numGpus, 1)) ;
%   for f = setdiff(fieldnames(stats)', {'num', 'time'})
%     f = char(f) ;
%     fprintf(' %s:', f) ;
%     fprintf(' %.3f', stats.(f)) ;
%   end
  format long
  fprintf('%s Iter %d Image %d: %.1f Hz, ', mode, numel(subset)*(state.epoch-1) + t, t, stats.num/stats.time * max(numGpus, 1));
  for kkk = 1:numel(rst)
     fprintf('%s = %.4f, ', rst(kkk).blob_name, rst(kkk).data); 
  end
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;
%end
%

%  ========liu@0908 added ================
function rst = check_error(net_vars, use_conv4)
    %rst = struct([]);
    if use_conv4
        acc_idx = 31;
        bbox_idx = 34;
        cls_idx = 30;
        label_idx = 29;
        cls_pred_idx = 27;
    else
        acc_idx = 38;  % 38 --> 32
        bbox_idx = 41; % 41 --> 35
        cls_idx = 37;  % 37 --> 31
        label_idx = 36;
        cls_pred_idx = 34;
    end
    % accuracy
    rst(1) = struct('blob_name', net_vars(acc_idx).name, 'data', gather(net_vars(acc_idx).value));
    % bbox loss
    rst(end + 1) = struct('blob_name', net_vars(bbox_idx).name, 'data', gather(net_vars(bbox_idx).value));
    % cls loss
    rst(end + 1) = struct('blob_name', net_vars(cls_idx).name, 'data', gather(net_vars(cls_idx).value));
    % fg accuracy and bg accuracy
    labels = gather(net_vars(label_idx).value(:,:,1));  %36 --> 29
    labels_weights = gather(net_vars(label_idx).value(:,:,2));  %36 --> 29
    
    %both prediction and gt are faces
    tmp_pred = gather(net_vars(cls_pred_idx).value);  %34 --> 27
    tmp_pred = reshape(tmp_pred, size(tmp_pred,1), [], 2);
    acc_fg = (tmp_pred(:,:,1) < tmp_pred(:,:,2)) & (labels == 1);
    %both prediction and gt are nonfaces
    acc_bg = (tmp_pred(:,:,1) >= tmp_pred(:,:,2)) & (labels == 2);
    
    accy_fg = sum(acc_fg(:) .* labels_weights(:)) / sum(labels_weights(labels == 1)+eps);
    accy_bg = sum(acc_bg(:) .* labels_weights(:)) / sum(labels_weights(labels == 2)+eps);
    
    rst(end + 1) = struct('blob_name', 'accuracy_fg', 'data', accy_fg);
    rst(end + 1) = struct('blob_name', 'accuracy_bg', 'data', accy_bg);
%end

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)

  % bring in gradients from other GPUs if any
  if ~isempty(mmap)
    numGpus = numel(mmap.Data) ;
    tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
    for g = setdiff(1:numGpus, labindex)
      tmp = tmp + mmap.Data(g).(net.params(p).name) ;
    end
    net.params(p).der = net.params(p).der + tmp ;
  else
    numGpus = 1 ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;

    case 'gradient'
        thisDecay = opts.weightDecay * net.params(p).weightDecay;
        thisLR = state.learningRate * net.params(p).learningRate;
        state.momentum{p} = opts.momentum * state.momentum{p} ...
                            + thisLR * (net.params(p).der + thisDecay * net.params(p).value);
        net.params(p).value = net.params(p).value - state.momentum{p};
%       thisDecay = opts.weightDecay * net.params(p).weightDecay ;
%       thisLR = state.learningRate * net.params(p).learningRate ;
%       state.momentum{p} = opts.momentum * state.momentum{p} ...
%         - thisDecay * net.params(p).value ...
%         - (1 / batchSize) * net.params(p).der;% (1 / batchSize/ opts.batchPerImage) * net.params(p).der
%       net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel1 = find(cellfun(@(x) isa(x,'SmoothL1Loss'), {net.layers.block})) ;
sel2 = find(cellfun(@(x) isa(x,'ReshapeSoftmaxLoss'), {net.layers.block})) ;
sel3 = find(cellfun(@(x) isa(x,'ReshapeAccuracy'), {net.layers.block})) ;
sel = [sel1 sel2 sel3];
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
