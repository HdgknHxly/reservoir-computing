classdef ESN < handle
    properties
        Nu         % Input dimension
        Nx         % Number of nodes in the reservoir
        Ny         % Output dimension
        Nl         % Number of reservoir layers
        Input
        Reservoir
        Inter
        Output
    end
    
    methods
        % Constructor
        function obj = ESN(Nu, Nx, Ny, Nl, inputScale, networkDensity, spectralRadius, leakRate, interScale)
            obj.Nu = Nu;
            obj.Nx = Nx;
            obj.Ny = Ny;
            obj.Nl = Nl;
            obj.Input = Input(Nu, Nx, inputScale);
            obj.Reservoir = Reservoir(Nx, Nl, networkDensity, spectralRadius, leakRate);
            obj.Inter = Inter(Nx, Nl, interScale);
            obj.Output = Output(Nx, Ny, Nl);
        end

        % Batch learning
        function WoutOpt = train(obj, UTrain, DTrain, optimizer)
            X = zeros(obj.Nx, obj.Nl);
            for i = 1:size(UTrain,1)
                u = UTrain(i,:)';
                for iLayer = 1:obj.Nl
                    if (iLayer == 1)
                        x0 = [obj.Input(u); iLayer];
                    else
                        x0 = [obj.Inter([X(:,iLayer-1); iLayer-1]); iLayer];
                    end
                    X(:,iLayer) = obj.Reservoir(x0);
                end
                x = reshape(X, [], 1);

                % Target data
                d = DTrain(i,:)';

                % Optimizer
                optimizer.update(x, d);
            end
            % Set trained Wout.
            WoutOpt = optimizer.getWoutOpt();
            obj.Output.setWoutOpt(WoutOpt);
        end

        % Prediction after batch learning
        function [XPred, YPred] = predict(obj, UTest)
            YPred = zeros(size(UTest,1), obj.Ny);
            XPred = zeros(size(UTest,1), obj.Nx*obj.Nl);
            X = zeros(obj.Nx, obj.Nl);
            for i = 1:size(UTest, 1)
                u = UTest(i,:)';
                for iLayer = 1:obj.Nl
                    if (iLayer == 1)
                        x0 = [obj.Input(u); iLayer];
                    else
                        x0 = [obj.Inter([X(:,iLayer-1); iLayer-1]); iLayer];
                    end
                    X(:,iLayer) = obj.Reservoir(x0);
                end
                x = reshape(X, [], 1);
                XPred(i,:) = x';

                % Model output after training
                yPred = obj.Output(x);
                YPred(i,:) = yPred';
            end
        end

        % Prediction after batch learning (free run)
        function [XRun, YRun] = run(obj, UTest)
            YRun = zeros(size(UTest,1), obj.Ny);
            XRun = zeros(size(UTest,1), obj.Nx*obj.Nl);
            u = UTest(1,:)';
            X = zeros(obj.Nx, obj.Nl);
            for i = 1:size(UTest,1)
                if (i > 1)
                    u(1:obj.Ny) = [];
                    u = [u; yRun'];
                end
                for iLayer = 1:obj.Nl
                    if (iLayer == 1)
                        x0 = [obj.Input(u); iLayer];
                    else
                        x0 = [obj.Inter([X(:,iLayer-1); iLayer-1]); iLayer];
                    end
                    X(:,iLayer) = obj.Reservoir(x0);
                end
                x = reshape(X, [], 1);
                XRun(i,:) = x'; 

                % Model output after training
                yRun = obj.Output(x);
                YRun(i,:) = yRun';
            end
        end
    end
end
