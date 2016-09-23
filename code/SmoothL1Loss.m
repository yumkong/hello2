classdef SmoothL1Loss < dagnn.ElementWise
    properties
        loss = 'smoothl1'
        opts = {}
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            %mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
            % prediction - gt
            % see fast rcnn paper for the equation
            coordinate_diff = inputs{1} - inputs{2};
            % liu@0812: multiply bbox_loss
            coordinate_diff = coordinate_diff .* inputs{3};

            pos_idx = (abs(coordinate_diff) < 1);
            pos_sum = sum(0.5 * (coordinate_diff(pos_idx).^2));
            neg_sum = sum(abs(coordinate_diff(~pos_idx)) - 0.5);
            %0902 changed
            %outputs{1} = pos_sum + neg_sum;
            outputs{1} = (pos_sum + neg_sum)/size(inputs{1},1)/size(inputs{1},2);

            %liu@0812: 
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        %       mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
        %       derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, ...
        %                                'loss', obj.loss, ...
        %                                'instanceWeights', 1./mass) ;
            tmpRes = gpuArray(zeros(size(inputs{1})));
            coordinate_diff = inputs{1} - inputs{2};
            % see fast rcnn paper for the equation
            mid_idx = (abs(coordinate_diff) < 1);
            left_idx = (coordinate_diff <= -1);
            right_idx = (coordinate_diff >= 1);
            tmpRes(mid_idx) = coordinate_diff(mid_idx);
            tmpRes(left_idx) = gpuArray(-1);
            tmpRes(right_idx) = gpuArray(1);
            % liu@0812: multiply bbox_loss
            %0902 changed
            %tmpRes = tmpRes .* inputs{3}  / (sum(inputs{3}(:)) + 1e-4) * 4;
            tmpRes = derOutputs{1} * tmpRes .* inputs{3} / size(inputs{1},1)/size(inputs{1},2);

            derInputs{1} = tmpRes;

          derInputs{2} = [] ;
          derInputs{3} = [] ;
          derParams = {} ;
        end

        function obj = SmoothL1Loss(varargin)
          obj.load(varargin) ;
          %liu@0809: make sure loss type is 'smoothl1'
          assert(strcmp(obj.loss, 'smoothl1'));
        end
    end
end
