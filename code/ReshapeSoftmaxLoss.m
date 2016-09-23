classdef ReshapeSoftmaxLoss < dagnn.ElementWise
  properties
        anchorNum = 7 %5 -- test, 9 -- train
        loss = 'softmaxlog'
        opts = {}
  end
  
  properties (Transient)
        average = 0
        numAveraged = 0
  end
    
  methods
    function outputs = forward(self, inputs, params)
        %liu@0811: added reshape functionality
        [w_, h_, ~] = size(inputs{1});
        input_reshaped = reshape(inputs{1}, [w_ h_*self.anchorNum 2]);
        %outputs{1} = vl_nnsoftmax(input_reshaped);
        
        E = exp(bsxfun(@minus, input_reshaped, max(input_reshaped,[],3))) ;
        L = sum(E, 3);
        softmaxY = bsxfun(@rdivide, E, L);
        inputSize = [size(softmaxY,1) size(softmaxY,2) size(softmaxY,3) size(softmaxY,4)];
        c = inputs{2}; %label
        
        %assert(isequal(labelSize(1:2), inputSize(1:2)));
        %assert(labelSize(4) == inputSize(4));
        % (label) or (label + label_weights)
        %assert(labelSize(3) == 1 | labelSize(3) == 2);

        % class c = 0 skips a spatial location
        mass = single(c(:,:,1,:) > 0);
        if size(c, 3) == 2
            % the second channel of c (if present) is used as weight
            mass = mass .* c(:,:,2,:);
            c(:,:,2,:) = [] ;
        end
        labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)];
        numPixelsPerImage = prod(inputSize(1:2)) ;
        numPixels = numPixelsPerImage * inputSize(4) ;
        imageVolume = numPixelsPerImage * inputSize(3) ;

        n = reshape(0:numPixels-1,labelSize) ;
        offset = 1 + mod(n, numPixelsPerImage) + ...
                 imageVolume * fix(n / numPixelsPerImage);
        % set the first channel as nonface score, the second channel as
        % face score
        c_rev = c;
        c_rev(c == 1) = 2;
        c_rev(c == 2) = 1;
        %ci = offset + numPixelsPerImage * max(c - 1,0);
        ci = offset + numPixelsPerImage * max(c_rev - 1,0);
        t = -log(softmaxY(ci)+eps);
        %0408 added '/ ***', divide m (# training examples)
        outputs{1} = mass(:)' * t(:) / (sum(mass(:))+ eps);
%         if nargin <= 1, return ; end
%         % backward
%         Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, 3)) ;

        %liu@0812: 
        n = self.numAveraged ;
        m = n + size(inputs{1},4) ;
        self.average = (n * self.average + double(gather(outputs{1}))) / m ;
        self.numAveraged = m;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        %liu@0811: added reshape functionality
        [w_, h_, ~] = size(inputs{1});
        input_reshaped = reshape(inputs{1}, [w_ h_*self.anchorNum 2]);
        %outputs{1} = vl_nnsoftmax(input_reshaped);
        
        E = exp(bsxfun(@minus, input_reshaped, max(input_reshaped,[],3))) ;
        L = sum(E, 3);
        softmaxY = bsxfun(@rdivide, E, L);
        c = inputs{2}; %label
        label = c(:,:,1);
        mask = c(:,:,2);
        tosubtract1 = single(label == 1);
        tosubtract2 = single(label == 2);
        %tosubtract = cat(3, tosubtract1, tosubtract2);
        % set the first channel as nonface score, the second channel as
        % face score
        tosubtract = cat(3, tosubtract2, tosubtract1);
        
        %derInputs_reshape = bsxfun(@times, (softmaxY - tosubtract), mask) * derOutputs{1} / (sum(mask(:))+ eps);
        sum_mask = sum(mask(:));
        if sum_mask ~= 0
            derInputs_reshape = bsxfun(@times, (softmaxY - tosubtract), mask) * derOutputs{1} / sum(mask(:));
        else
            derInputs_reshape = zeros(size(softmaxY), 'single');
        end
        %derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
        %derInputs_reshape = vl_nnsoftmax(input_reshaped, derOutputs{1});
        derInputs{1} = reshape(derInputs_reshape, [w_ h_ 2*self.anchorNum]);
        derInputs{2} = [];
        derParams = {} ;
    end

    function obj = ReshapeSoftmaxLoss(varargin)
      obj.load(varargin) ;
    end
  end
end