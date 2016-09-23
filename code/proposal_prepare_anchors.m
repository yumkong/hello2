function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    map_name = 'output_map.mat';
    try 
        load(map_name);
    catch
        [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
        save(map_name, 'output_width_map','output_height_map');
    end
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[-1:5]); %[3:5] --> [0:5]
end

function [output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file)
% --------------------------------------------------------

    net = vl_simplenn_tidy(load(test_net_def_file)) ;
    net.layers = net.layers(1:30) ;
    % Convert the model from SimpleNN to DagNN
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.conserveMemory = true;
    
    input = 100:conf.max_size;
    output_w = nan(size(input));
    output_h = nan(size(input));
    for i = 1:length(input)
        fprintf('Processing image %d / %d\n',i, length(input));
        s = input(i);
        im_blob = single(zeros(s, s, 3, 1));
        net.eval({'input', im_blob});

        conv5_3_relu_res = net.vars(net.getVarIndex('x30')).value;
        output_w(i) = size(conv5_3_relu_res, 1);
        output_h(i) = size(conv5_3_relu_res, 2);
    end
    
    output_width_map = containers.Map(input, output_w);
    output_height_map = containers.Map(input, output_h);
    
end