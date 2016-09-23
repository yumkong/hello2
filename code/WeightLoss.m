classdef WeightLoss < dagnn.ElementWise
    properties
        loss = 'log'
        opts = {}
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
    end
  
    methods
        function outputs = forward(self, inputs, params)
            %liu@0811: add the third argument 
            %outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
            outputs{1} = vl_nnloss2(inputs{1}, inputs{2}, [], 'loss', self.loss, self.opts{:}) ;
            n = self.numAveraged ;
            m = n + size(inputs{1},4) ;
            self.average = (n * self.average + gather(outputs{1})) / m ;
            self.numAveraged = m ;
        end

        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            %liu@0811: add the third argument
            derInputs{1} = vl_nnloss2(inputs{1}, inputs{2}, derOutputs{1}, 'loss', self.loss, self.opts{:});
            derInputs{2} = [];
            derParams = {};
        end

        function obj = WeightLoss(varargin)
            obj.load(varargin) ;
        end
    end
end