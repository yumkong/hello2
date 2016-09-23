function main_train_rpn()
%%% ---[ 1. First Contact with the Layer system ]--- %%%
clear
clc
run ../../../matlab/vl_setupnn;

% prepare widerface training and test data
use_flipped  = false;
dataset      = [];
dataset      = widerface_all(dataset, 'train', use_flipped);
dataset      = widerface_all(dataset, 'test', false);

%load proposal configuration file
%0921 added
use_conv4 = true;

if use_conv4
    conf_proposal = proposal_config_widerface('feat_stride',8, 'test_min_box_size',8);
else
    conf_proposal = proposal_config_widerface('feat_stride',16, 'test_min_box_size',16);
end
% generate anchors and pre-calculate output size of rpn network 
model.rpn_cache_name = 'rpn_cache';
model.premodel_file = '../data/models/imagenet-vgg-verydeep-16.mat';

%20160510 added, gpu switch
use_gpu = true;

if use_gpu
    gpuID = [1];
else
    gpuID = [];
end 

if use_conv4
    [conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
         = proposal_prepare_anchors_conv4(conf_proposal, model.rpn_cache_name, model.premodel_file, gpuID);
     roi_database_name = '../data_conv4/roidb_conv4_package.mat'; %previous: 'roidb_package.mat'
else 
    [conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
         = proposal_prepare_anchors(conf_proposal, model.rpn_cache_name, model.premodel_file);
    roi_database_name = '../data_conv5/roidb_package.mat'; %previous: 'roidb_package.mat'
end

try
    load(roi_database_name); %contains: image_roidb_train,image_roidb_val,bbox_means,bbox_stds
catch
    fprintf('Preparing training data...');
    [image_roidb_train, bbox_means, bbox_stds]...
                        = proposal_prepare_image_roidb(conf_proposal, dataset.imdb_train, dataset.roidb_train);
    fprintf('Done.\n');                     

    fprintf('Preparing validation data...');
    image_roidb_val = proposal_prepare_image_roidb(conf_proposal, dataset.imdb_test, dataset.roidb_test, bbox_means, bbox_stds);
    fprintf('Done.\n');
    save(roi_database_name,'image_roidb_train','image_roidb_val','bbox_means','bbox_stds');
end

% liu@0809: create data blob
imdbWider.data = cat(1, image_roidb_train, image_roidb_val);
imdbWider.set = cat(1, ones(length(image_roidb_train), 1), 2*ones(length(image_roidb_val), 1));

%initialize rpn network
if use_conv4
    net = rpnInitializeModel_conv4();
else
    net = rpnInitializeModel();
end

%liu@0809: mean image data
conf_proposal.mean_image = single(net.meta.normalization.averageImage);  %1x1x3 array
conf_proposal.image_means = single(net.meta.normalization.averageImage);


% liu@0808: change 2nd input arg from function name (string) to function handle for easy debug
%launch_net(imdbWider, 'wider_clfnet_30x30','res_0808', gpuID, net);
trainer_handle = @train_ClfReg;
launch_net(conf_proposal, imdbWider, trainer_handle,'../train_results/res_0922_conv4_t1', gpuID, use_conv4, net);
end

function outdata = prepare_trainval_data(conf, indata)
     %parfor i = 1:length(indata)
     %celldata = struct2cell(indata);
     num_images = conf.ims_per_batch;
     rois_per_image = conf.batch_size / num_images;
     fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
     
     total_num = length(indata);
     outdata1 = cell(total_num, 1);
     for i = 1:total_num
%         if i == 280
%           fprintf('watch here.\n');
%         end
%         [labels, label_weights, bbox_targets, bbox_loss] = sample_rois(conf, ...
%                     imdb_single, fg_rois_per_image, rois_per_image, im_scale);
        image_roidb = indata(i);
        bbox_targets = image_roidb.bbox_targets{1};
        ex_asign_labels = bbox_targets(:, 1);

        % Select foreground ROIs as those with >= FG_THRESH overlap
        % 0822 correct error
        %fg_inds = find(bbox_targets(:, 1) > conf.fg_thresh);  % > 0
        fg_inds = find(bbox_targets(:, 1) > 0);  % > 0

        % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        %bg_inds = find((bbox_targets(:, 1) <= conf.bg_thresh_hi) & (bbox_targets(:, 1) >= conf.bg_thresh_lo));
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

        bbox_loss = bbox_targets * 0;
        bbox_loss(fg_inds, :) = 1;  % [1 1 1 1] for all fg items

        % get fcn output size  from rescaled input image
        img_size = round(image_roidb.im_size * image_roidb.image_scale);
        output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);

        %assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));
        % liu@0802: resize to [9, conv5_height, conv5_width]
        % liu@0802: only fgs positions are labeled as 1
        labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
        % liu@0802: resize to [9, conv5_height, conv5_width]
        % liu@0802: fgs positions + selected bgs (fgs+ selected bgs = 256) are labeled as 1
        label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
        % liu@0802: resize to [4x7, conv5_height, conv5_width]
        bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
        bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));

        % permute from [channel, height, width], where channel is the
        % fastest dimension to [width, height, channel]
        % liu@0802: permute to [conv5_height, conv5_width, 9]
        labels_blob = permute(labels_blob, [2, 3, 1]);
        %***liu@0811: reshape to directly fit for loss function
        [h_, w_, c_] = size(labels_blob);
        labels_blob = reshape(labels_blob, [h_, w_*c_, 1]);

        % liu@0802: permute to [conv5_height, conv5_width, 9]
        label_weights_blob = permute(label_weights_blob, [2, 3, 1]);
        %***liu@0811: reshape to directly fit for loss function
        [h_, w_, c_] = size(label_weights_blob);
        label_weights_blob = reshape(label_weights_blob, [h_, w_*c_, 1]);

        % liu@0802: permute to [conv5_height, conv5_width, 4x7]
        bbox_targets_blob = permute(bbox_targets_blob, [2, 3, 1]);
        % liu@0802: permute to [conv5_height, conv5_width, 9x4]
        bbox_loss_blob = permute(bbox_loss_blob, [2, 3, 1]);

        labels_blob = single(labels_blob);
        % liu@0814: masked
        %labels_blob(labels_blob > 0) = 1; %to binary label (fg and bg)
        label_weights_blob = single(label_weights_blob);
        bbox_targets_blob = single(bbox_targets_blob); 
        bbox_loss_blob = single(bbox_loss_blob);

        %assert(~isempty(im_blob));
        assert(~isempty(labels_blob));
        assert(~isempty(label_weights_blob));
        assert(~isempty(bbox_targets_blob));
        assert(~isempty(bbox_loss_blob));

        % combine label and label weights
        label_all = cat(3, labels_blob, label_weights_blob);
        
        outdata1{i} = {image_roidb.image_path, image_roidb.image_scale, label_all, bbox_targets_blob, bbox_loss_blob};
     end
     fieldname = {'image_path','image_scale','label_all', 'bbox_targets_blob', 'bbox_loss_blob'};
     aa = [outdata1{:}];
     aa = reshape(aa, [], 5);
     aa = [outdata1{:}];
     aa = reshape(aa, 5, []);
     aa = aa';
     outdata = cell2struct(aa, fieldname, 2);
end

% % Generate a random sample of ROIs comprising foreground and background examples.
% function [labels, label_weights, bbox_targets, bbox_loss_weights] = sample_rois(conf, ...
%                             image_roidb, fg_rois_per_image, rois_per_image, im_scale)
% 
%     bbox_targets = image_roidb.bbox_targets{1};
%     ex_asign_labels = bbox_targets(:, 1);
%     
%     % Select foreground ROIs as those with >= FG_THRESH overlap
%     fg_inds = find(bbox_targets(:, 1) > conf.fg_thresh);  % > 0
%     
%     % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
%     bg_inds = find((bbox_targets(:, 1) <= conf.bg_thresh_hi) & (bbox_targets(:, 1) >= conf.bg_thresh_lo));
%     
%     % select foreground
%     fg_num = min(fg_rois_per_image, length(fg_inds));
%     fg_inds = fg_inds(randperm(length(fg_inds), fg_num)); % ramdomly select fg_num out of all fgs
%     
%     bg_num = min(rois_per_image - fg_num, length(bg_inds));
%     bg_inds = bg_inds(randperm(length(bg_inds), bg_num)); % ramdomly select bg_num out of all bgs
% 
%     %liu@0812 added: nonface -- 2
%     labels = 2*ones(size(bbox_targets, 1), 1);
%     % set foreground labels
%     labels(fg_inds) = ex_asign_labels(fg_inds);
%     %liu@0812 added: nonface -- 2
%     %labels(~fg_inds) = 2;
%     
%     assert(all(ex_asign_labels(fg_inds) > 0));
%     
%     label_weights = zeros(size(bbox_targets, 1), 1);
%     % set foreground labels weights
%     label_weights(fg_inds) = 1;
%     % set background labels weights
%     label_weights(bg_inds) = conf.bg_weight;
%     
%     bbox_targets = single(full(bbox_targets(:, 2:end)));
%     
%     bbox_loss_weights = bbox_targets * 0;
%     bbox_loss_weights(fg_inds, :) = 1;  % [1 1 1 1] for all fg items
% end