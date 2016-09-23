function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors_conv4(conf, cache_name, test_net_def_file, gpuID)
    map_name = '../data_conv4/output_map_conv4.mat';
    try 
        load(map_name);
    catch
        [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file, gpuID);
        save(map_name, 'output_width_map','output_height_map');
    end
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[-1:5]); %[3:5] --> [0:5]
end

function [output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file, gpuID)
% --------------------------------------------------------

    net = vl_simplenn_tidy(load(test_net_def_file));
    net.layers = net.layers(1:23);
    % Convert the model from SimpleNN to DagNN
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
    net.conserveMemory = true;
    
    %use gpu
    numGpus = numel(gpuID) ;
    if numGpus >= 1
      net.move('gpu');
    end
    
    input = 301:conf.max_size;
    output_w = nan(size(input));
    output_h = nan(size(input));
    for i = 1:length(input)
        fprintf('Processing image %d / %d\n',i, length(input));
        s = input(i);
        if numGpus >= 1
            im_blob = gpuArray(zeros(s, s, 3, 1, 'single'));
        else
            im_blob = zeros(s, s, 3, 1, 'single');
        end
        
        net.eval({'input', im_blob});

        if numGpus >= 1
            conv4_3_relu_res = gather(net.vars(net.getVarIndex('x23')).value);
        else
            conv4_3_relu_res = net.vars(net.getVarIndex('x23')).value;
        end
        output_h(i) = size(conv4_3_relu_res, 1);
        output_w(i) = size(conv4_3_relu_res, 2);
    end  
    
    output_height_map = containers.Map(input, output_h);
    output_width_map = containers.Map(input, output_w);
    
end