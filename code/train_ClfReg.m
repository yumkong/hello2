function [net, info] = train_ClfReg(conf, imdb, net, inpt, varargin)

    % some common options
    trainer = @cnn_train_dag_fasterRCNN;  %need to change
    %20160508 changed
    opts.train.batchSize = 1; 
    opts.train.batchPerImage = 256; % 
    %0510 make it smaller
    opts.train.numEpochs = 30;  %45 %60
    opts.train.continue = true ;
    %0510 changed
    %opts.train.gpus = [] ;
    opts.train.gpus = inpt.gpus ;

    %clf
    opts.train.learningRate = [1e-3*ones(1,80) 1e-4*ones(1,20)];  % caffe lr starts from 1e-3 
    opts.train.weightDecay = 5 * 1e-4;  % caffe weight decay 0.0005 
    opts.train.momentum = 0.9;
    opts.train.expDir = inpt.expDir;
    %0921 added
    opts.train.use_conv4 = inpt.use_conv4;
    
    %opts.train.savePlots = false;
    %liu@0810: no need to use it, since only one image/ batch
    %opts.train.numSubBatches = 2; % keep this, each subbatch 10/5=2 images
    % getBatch options
    bopts.useGpu = numel(opts.train.gpus) >  0 ;
    
    
    % -- end of the network

    % do the training!
    %initNet(net, 1e-2*ones(1, 5), 1e-2*ones(1, 5));
    net.conserveMemory = true;  %set true to save memory

    %info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
    trainData = find(imdb.set == 1);
    valData = find(imdb.set == 2);
    info = trainer(net, imdb, @(i) getBatchDisk(conf, bopts, i), opts.train, 'train', trainData, 'val', valData) ;
end

% getBatch for IMDBs that are too big to be in RAM
function inputs = getBatchDisk(conf, opts, imdb_single)
    % =============== add random seed =================
    %rng_seed = 6;
    %prev_rng = rng;  %keep previous random state
    %rng(rng_seed, 'twister');
  % =================================================

    num_images = length(imdb_single);
    assert(num_images == 1, 'getBatchDisk only support num_images == 1');
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scale] = get_image_blob(conf, imdb_single);
    
    [labels, label_weights, bbox_targets, bbox_loss] = sample_rois(conf, ...
                    imdb_single, fg_rois_per_image, rois_per_image, im_scale);
    
    % get fcn output size
    img_size = round(imdb_single(1).im_size * im_scale);
    output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
    %liu@0912
    %output_size = output_size(end:-1:1);
    img_size = img_size(end:-1:1);
    
    assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));
    % liu@0802: resize to [9, conv5_height, conv5_width]
    % liu@0802: only fgs positions are labeled as 1
    labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
    % liu@0802: resize to [9, conv5_height, conv5_width]
    % liu@0802: fgs positions + selected bgs (fgs+ selected bgs = 256) are labeled as 1
    label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
    % liu@0802: resize to [9x4, conv5_height, conv5_width]
    bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
    bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));
    
    % permute from [channel, height, width], where channel is the
    % fastest dimension to [width, height, channel]
    % liu@0802: permute to [conv5_height, conv5_width, 9]
    % liu@0912: permute to [conv5_width, conv5_height, 9]
    labels_blob = permute(labels_blob, [3, 2, 1]);
    %***liu@0811: reshape to directly fit for loss function
    %[h_, w_, c_] = size(labels_blob);
    %labels_blob = reshape(labels_blob, [h_, w_*c_, 1]);
    [w_, h_, c_] = size(labels_blob);
    labels_blob = reshape(labels_blob, [w_, h_*c_, 1]);
    
    % liu@0802: permute to [conv5_height, conv5_width, 9]
    % liu@0912: permute to [conv5_width, conv5_height, 9]
    label_weights_blob = permute(label_weights_blob, [3, 2, 1]);
    %***liu@0811: reshape to directly fit for loss function
    [w_, h_, c_] = size(label_weights_blob);
    label_weights_blob = reshape(label_weights_blob, [w_, h_*c_, 1]);
    
    % liu@0802: permute to [conv5_height, conv5_width, 9x4]
    % liu@0912: permute to [conv5_width, conv5_height, 9x4]
    bbox_targets_blob = permute(bbox_targets_blob, [3, 2, 1]);
    % liu@0802: permute to [conv5_height, conv5_width, 9x4]
    % liu@0912: permute to [conv5_width, conv5_height, 9x4]
    bbox_loss_blob = permute(bbox_loss_blob, [3, 2, 1]);
    
    labels_blob = single(labels_blob);
    %labels_blob(labels_blob > 0) = 1; %to binary label (fg and bg)
    label_weights_blob = single(label_weights_blob);
    bbox_targets_blob = single(bbox_targets_blob); 
    bbox_loss_blob = single(bbox_loss_blob);
    
    assert(~isempty(im_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(label_weights_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
    
    % combine label and label weights
    label_all = cat(3, labels_blob, label_weights_blob);
    % data transformation has already been done in main function
	if opts.useGpu > 0
  		im_blob = gpuArray(im_blob);
        label_all = gpuArray(label_all);
        %labels_blob = gpuArray(labels_blob);
        %label_weights_blob = gpuArray(label_weights_blob);
        bbox_targets_blob = gpuArray(bbox_targets_blob);
        bbox_loss_blob = gpuArray(bbox_loss_blob);        
	end
    
    inputs = {'input',im_blob, 'label',label_all, 'bbox_targets',bbox_targets_blob, 'bbox_loss', bbox_loss_blob};
    
    % =============== recover previous random seed =================
    %rng(prev_rng);
  % ==============================================================
end

% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scale] = get_image_blob(conf, images)
    im = imread(images(1).image_path);
    %im = vl_imreadjpeg({images(1).image_path}); %'numThreads', 1
    %im = im{1};
    target_size = conf.scales(1);
    %[im_blob, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
    % directly use the content of prep_im_for_blob
    im = single(im);
    % liu@0908: found and corrected an error: 
    % conf.image_means ==> conf.mean_image
    im = bsxfun(@minus, im, single(conf.mean_image));   
    % rescale to the target size
    im_scale = prep_im_for_blob_size(size(im), target_size, conf.max_size);
    im_size = round([size(im,1) size(im,2)] * im_scale);
    im_blob = imresize(im, im_size, 'bilinear', 'antialiasing', false);
    
    im_blob = im_blob(:,:, [3,2,1], :);
    im_blob = permute(im_blob, [2,1,3,4]);
end

% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, label_weights, bbox_targets, bbox_loss_weights] = sample_rois(conf, ...
                            image_roidb, fg_rois_per_image, rois_per_image, im_scale)

    bbox_targets = image_roidb.bbox_targets{1};
    ex_asign_labels = bbox_targets(:, 1);
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    % ***liu@0822*** found an error here, correct it as follows
    %fg_inds = find(bbox_targets(:, 1) > conf.fg_thresh);
    fg_inds = find(bbox_targets(:, 1) > 0);
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    % ***liu@0822*** found an error here, correct it as follows
    % bg_inds = find((bbox_targets(:, 1) <= conf.bg_thresh_hi) & (bbox_targets(:, 1) >= conf.bg_thresh_lo));
    bg_inds = find(bbox_targets(:, 1) < 0);
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num)); % ramdomly select fg_num out of all fgs
    
    bg_num = min(rois_per_image - fg_num, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num)); % ramdomly select bg_num out of all bgs

    %liu@0812 added: nonface -- 2
    labels = 2*ones(size(bbox_targets, 1), 1);
    % set foreground labels
    labels(fg_inds) = ex_asign_labels(fg_inds);
    %liu@0812 added: nonface -- 2
    %labels(~fg_inds) = 2;
    
    assert(all(ex_asign_labels(fg_inds) > 0));
    
    label_weights = zeros(size(bbox_targets, 1), 1);
    % set foreground labels weights
    label_weights(fg_inds) = 1;
    % set background labels weights
    label_weights(bg_inds) = conf.bg_weight;
    
    bbox_targets = single(full(bbox_targets(:, 2:end)));
    
    bbox_loss_weights = bbox_targets * 0;
    bbox_loss_weights(fg_inds, :) = 1;  % [1 1 1 1] for all fg items
end
