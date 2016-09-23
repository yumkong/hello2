function singleimg_test_conv5()

clear
clc

% use fullfile to flexibly adapt to linux and windows path conventions
addpath(fullfile('..','assistfunc'));
addpath(fullfile('..', 'export_fig'));
addpath(fullfile('..', 'code'));
%addpath('nms');

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','..', '..', 'matlab', 'vl_setupnn.m')) ;

%load('res_0917/new-net-epoch-30.mat');  %net
%load('res_0918/net-epoch-7.mat');  %net
load(fullfile('..','train_results','puck_conv5','net-epoch-30.mat'));  %net
save_suffix = 'puck_conv5_ep30'; % change it according to above
%box configuration
opts.per_nms_topN           = 300;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 50;
opts.use_gpu                = false;

conf = load(fullfile('..','data_conv5','output_map.mat'));
conf.anchor_num = 7;  % can change any time
conf.feat_stride = 16; %
%conf.anchors = proposal_generate_anchors('test_anchors.mat', 'ratios', [1], 'scales', 2.^[1:5]);
conf.anchors = proposal_generate_anchors('test_anchor', 'ratios', [1], 'scales', 2.^[-1:5]); %anchor6

%transform from struct to DagNN
net = dagnn.DagNN.loadobj(net);

net.removeLayer('loss_bbox'); % remove net.vars and net.layers together
net.removeLayer('accuracy');
net.removeLayer('loss_cls');

% to correct bbox regression parameters
load(fullfile('..','data_conv5','bbox_stat.mat')); %bbox_means, bbox_stds
anchor_size = size(conf.anchors, 1);
bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);

weights = net.params(31).value;
biase = net.params(32).value;

weights = ...
    bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds; 
biase = ...
    biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

net.params(31).value = weights;
net.params(32).value = biase;

% -------------------------------------------------------------------------
% use the model to classify an image
% -------------------------------------------------------------------------
meanface = net.meta.normalization.averageImage;

%20160510 added, gpu switch
use_gpu = true;

if use_gpu
    gpuID = [1];
else
    gpuID = [];
end 

numGpus = numel(gpuID) ;
if numGpus >= 1
  net.move('gpu');
end

test_img_dir = fullfile('..', 'test_image');
test_res_dir = fullfile('..', 'test_result');
if ~exist(test_res_dir, 'dir')
   mkdir(test_res_dir); 
end
ims = dir(fullfile(test_img_dir, '*.jpg'));  %*/ delete the 77th image
for i = 1:numel(ims)
    fprintf('********** Processing image: %d/%d *********\n', i, length(ims));
    tic
    imname = ims(i).name;
    img = single(imread(fullfile(test_img_dir, imname)));

    [hei, wid, ~] = size(img);
    %7 x N x 4
    
    % if training samples have subtraction operation, should do this
    imPatches = bsxfun(@minus, img, single(meanface));
    imPatches = imPatches(:,:,[3,2,1],:);
    imPatches = permute(imPatches, [2,1,3]);
    if numGpus >= 1
        imPatches = gpuArray(imPatches);
    end
    %start detection
    tic
    net.conserveMemory = false;
    net.eval({'input', imPatches});
    fprintf('Processing 1 image costs %.3f seconds\n', toc);
    
    box_deltas = net.vars(net.getVarIndex('proposal_bbox_pred')).value;  % [w, h, 28]
    if numGpus >= 1
       box_deltas = gather(box_deltas); 
    end
    %featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    box_deltas = permute(box_deltas, [3, 2, 1]);
    box_deltas = reshape(box_deltas, 4, [])';
    
    anchors = calc_anchors([hei wid], conf);
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
    pred_boxes = clip_boxes(pred_boxes, size(img, 2), size(img, 1));
    % *** add anchors
    anchors_clip = clip_boxes(anchors, size(img, 2), size(img, 1));
    
    % show the classification result
    scores = net.vars(net.getVarIndex('proposal_cls_score')).value;
    % B = squeeze(A) returns an array B with the same elements as A but
    %     with all the singleton dimensions removed
    if numGpus >= 1
       scores = gather(scores); 
    end
	scores = squeeze(scores);
    % w x (h x 7) x 2
    scores = reshape(scores, size(scores,1), [], 2);
    E = exp(bsxfun(@minus, scores, max(scores,[],3))) ;
    L = sum(E, 3);
    scores = bsxfun(@rdivide, E, L);
    %[~, best] = max(scores,[],3);
    scores = scores(:,:,2);
    scores = reshape(scores, size(scores,1), [], 7); % w x h x 7
    scores = permute(scores, [3, 2, 1]); % 7 x h x w
    scores = scores(:);
    
    % drop too small boxes: less than 5 pixels
    [pred_boxes, scores, anchors_clip] = filter_boxes(5, pred_boxes, scores, anchors_clip); 
    
    % sort
    [scores, scores_ind] = sort(scores, 'descend');
    pred_boxes = pred_boxes(scores_ind, :);
    % *** add anchors
    anchors_clip = anchors_clip(scores_ind, :);
    
    %image(single(im)/255); 
    imshow(img/255); 
    axis image;
    axis off;
    set(gcf, 'Color', 'white');
    endNum = sum(scores >= 0.9);
    for j = 1:endNum  % can be changed to any positive number to show different #proposals
        bbox = pred_boxes(j,:);
        rect = [bbox(:, 1), bbox(:, 2), bbox(:, 3)-bbox(:,1)+1, bbox(:,4)-bbox(2)+1];
        rectangle('Position', rect, 'LineWidth', 1, 'EdgeColor', [0 1 0]);
        % *** add anchor
%         bbox_anchor = anchors_clip(j,:);
%         rect_anchor = [bbox_anchor(:, 1), bbox_anchor(:, 2), bbox_anchor(:, 3)-bbox_anchor(:,1)+1, bbox_anchor(:,4)-bbox_anchor(2)+1];
%         rectangle('Position', rect_anchor, 'LineWidth', 1, 'EdgeColor', [1 0 0]);
    end
    %saveName = sprintf('test_result/img_%d_score_0918_plus_anchor',i);
    saveName = fullfile(test_res_dir, sprintf('img_%d_%s',i,save_suffix));
    export_fig(saveName, '-png', '-a1', '-native');
    fprintf('image %d saved.\n', i);
end

end  %of main function

function anchors = calc_anchors(im_size, conf)

    %im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
    img_size = im_size;
    %img_size = round(im_size * im_scale);
    output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
   
    shift_x = [0:(output_size(2)-1)] * conf.feat_stride;
    shift_y = [0:(output_size(1)-1)] * conf.feat_stride;
    [shift_x, shift_y] = meshgrid(shift_x, shift_y);
    
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors = reshape(bsxfun(@plus, permute(conf.anchors, [1, 3, 2]), ...
        permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);
end

function bbox = calcu_bbox(ex_boxes, regression_label)
    %regression_label = [targets_dx, targets_dy, targets_dw, targets_dh];
    bbox = zeros(size(ex_boxes));
    
    targets_dx = regression_label(:, 1);
    targets_dy = regression_label(:, 2);
    targets_dw = regression_label(:, 3);
    targets_dh = regression_label(:, 4);
    
    ex_widths = ex_boxes(:, 3) - ex_boxes(:, 1) + 1;
    ex_heights = ex_boxes(:, 4) - ex_boxes(:, 2) + 1;
    ex_ctr_x = ex_boxes(:, 1) + 0.5 * (ex_widths - 1);
    ex_ctr_y = ex_boxes(:, 2) + 0.5 * (ex_heights - 1);
    
    gt_heights = exp(targets_dh) .* ex_heights;
    gt_widths = exp(targets_dw) .* ex_widths;
    gt_ctr_y = targets_dy .* ex_heights + ex_ctr_y;
    gt_ctr_x = targets_dx .* ex_widths + ex_ctr_x;
    
    bbox(:, 1) = gt_ctr_x - 0.5 * (gt_widths - 1);
    bbox(:, 2) = gt_ctr_y - 0.5 * (gt_heights - 1);
    bbox(:, 3) = gt_widths + bbox(:, 1) - 1;
    bbox(:, 4) = gt_heights + bbox(:, 2) - 1;
    
%     gt_widths = gt_boxes(:, 3) - gt_boxes(:, 1) + 1;
%     gt_heights = gt_boxes(:, 4) - gt_boxes(:, 2) + 1;
%     gt_ctr_x = gt_boxes(:, 1) + 0.5 * (gt_widths - 1);
%     gt_ctr_y = gt_boxes(:, 2) + 0.5 * (gt_heights - 1);
%     
%     targets_dx = (gt_ctr_x - ex_ctr_x) ./ (ex_widths+eps);
%     targets_dy = (gt_ctr_y - ex_ctr_y) ./ (ex_heights+eps);
%     targets_dw = log(gt_widths ./ ex_widths);
%     targets_dh = log(gt_heights ./ ex_heights); 
end

function [boxes, scores, anchors_clip] = filter_boxes(min_box_size, boxes, scores, anchors_clip)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
    % *** add anchor
    anchors_clip = anchors_clip(valid_ind, :);
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end