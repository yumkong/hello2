function main_test_rpn()

clear
clc

addpath('./assistfunc');
addpath('./export_fig');
addpath('nms');

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

load('res_0902/net-epoch-30.mat');  %net
%load('res_anchor6/net-epoch-5.mat');  %net

%box configuration
opts.per_nms_topN           = 300;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 50;
opts.use_gpu                = false;

conf = load('output_map.mat');
conf.anchor_num = 7;  % can change any time
conf.feat_stride = 16; %
%conf.anchors = proposal_generate_anchors('test_anchors.mat', 'ratios', [1], 'scales', 2.^[1:5]);
conf.anchors = proposal_generate_anchors('test_anchor', 'ratios', [1], 'scales', 2.^[-1:5]); %anchor6

%transform from struct to DagNN
net = dagnn.DagNN.loadobj(net);

net.removeLayer('loss_bbox'); % remove net.vars and net.layers together
net.removeLayer('accuracy');
net.removeLayer('loss_cls');
% net.vars(42) = []; % delete loss_bbox layer
% net.vars(39) = []; % delete loss_bbox layer
% net.vars(38) = []; % delete loss_bbox layer
% net.layers(end) = []; % delete loss_bbox layer
% net.layers(end) = []; % delete accuracy layer
% net.layers(end) = []; % delete loss_cls layer
% -------------------------------------------------------------------------
% use the model to classify an image
% -------------------------------------------------------------------------
meanface = net.meta.normalization.averageImage;

ims = dir('./test_image/*.jpg');  %*/ delete the 77th image
for i = 1:numel(ims)
    fprintf('********** Processing image: %d/%d *********\n', i, length(ims));
    tic
    imname = ims(i).name;
    img = single(imread(['./test_image/' imname]));

    [hei, wid, ~] = size(img);
    anchors = calc_anchors([hei wid], conf);
    % if training samples have subtraction operation, should do this
    imPatches = bsxfun(@minus, img, meanface);

    %start detection
    tic
    net.conserveMemory = true;
    net.eval({'input', imPatches});
    fprintf('Processing 1 image costs %.3f seconds\n', toc);
    % show the classification result
    scores = net.vars(net.getVarIndex('prediction')).value;
	scores = squeeze(gather(scores)); 
    [bestScore, best] = max(scores,[],3);

    siz1 = size(best, 1); %h
    %siz2 = size(best, 2);
    best_reshape = reshape(best, siz1, [], conf.anchor_num);
    % #anchor x hei x wid
    best_reshape = permute(best_reshape, [3, 1, 2]);
    
    %liu@0815
    score_reshape = reshape(bestScore, siz1, [], conf.anchor_num);
    % #anchor x hei x wid
    score_reshape = permute(score_reshape, [3, 1, 2]);
    
    bbox_reg = net.vars(net.getVarIndex('proposal_bbox_pred')).value;
    bbox_reg = squeeze(gather(bbox_reg)); 
    % reshape predicted bbox to be consistent with anchors
    
    % H x W x 28 --> 28 x H x W
    bbox_reg1 = permute(bbox_reg, [3, 1, 2]);
    % 4 x 7 x H x W
    bbox_reg_reshape = reshape(bbox_reg1, 4, [])';
    %bbox_reg_reshape = reshape(bbox_reg1, 4, conf.anchor_num, size(bbox_reg, 1), size(bbox_reg, 2));
    %bbox_reg_reshape = reshape(bbox_reg_reshape, 4, []);
    %bbox_reg_reshape = bbox_reg_reshape';%'
    
    face_idx = find(best_reshape == 1);
    % #estimated_face x 4
    face_reg = bbox_reg_reshape(face_idx, :);
    %
    face_anchors = anchors(face_idx, :);
    
    bbox = round(calcu_bbox(face_anchors, face_reg));
    
    bbox_score = score_reshape(face_idx);
    abbox = [bbox bbox_score];
    %sort in descending order
    [bbox_score, sort_idx] = sort(bbox_score, 1, 'descend');
    idx1 = sum(bbox_score >= 0.9);
    abbox = abbox(sort_idx, :);
    
    %tic
    aboxes = abbox(1:idx1,:);
    %aboxes = boxes_filter(abbox, opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    %fprintf('NMS costs %.3f seconds\n', toc);
    %aboxes = abbox;
%     tic
%     % processing method from NPD @PAMI15
%     rects = PostProcRects(inrec);
%     numFaces = length(rects);    
%     fprintf('%d faces detected.\n', numFaces);
%     fprintf('Postprocess time:%.2f seconds\n\n',toc);
    %img = uint8(img);
    numFaces = size(aboxes, 1);
    img_out = uint8(img);
    if numFaces > 0
        %border = round(size(img,2) / 300);
        %if border < 2, border = 2; end
        border = 2;
        for j = 1 : numFaces
            %img = DrawRectangle(img, rects(j).col, rects(j).row, rects(j).size, rects(j).size+2, [0 255 0], border);
            img_out = DrawRectangle(img_out, aboxes(j,1), aboxes(j,2), aboxes(j,3)-aboxes(j,1)+1, aboxes(j,4)-aboxes(j,2)+1, [0 255 0], border);
        end
    end

    saveName = sprintf('test_result/%s_score_0.9.jpg',imname(1:end-4));
    imwrite(img_out, saveName, 'jpg', 'Quality', 100);
    %export_fig(saveName, '-png', '-a1', '-native');
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

% for mex cpp debug
%mexPrintf("ok3\n");
%mexPrintf("feat6 dim 1 = %d\n", heiall[6]);
function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        % only Ç° keep per_nms_topN žöboxes,ŽóÔŒÎªscoreÖµ>=0.35µÄboxes
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end