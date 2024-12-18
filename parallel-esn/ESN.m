classdef ESN < handle
    properties
        Nu         % Input dimension
        Nx         % Number of nodes in the reservoir
        Ny         % Output dimension
        Input1
        Input2
        Reservoir1
        Reservoir2
        Output
    end
    
    methods
        function obj = ESN(Nu, Nx, Ny, inputScaling, networkDensity, spectralRadius, leakRate)
            obj.Nu = Nu;
            obj.Nx = Nx;
            obj.Ny = Ny;
            obj.Input1 = Input(Nu, Nx, inputScaling);
            obj.Input2 = Input(Nu, Nx, inputScaling);
            obj.Reservoir1 = Reservoir(Nx, networkDensity, spectralRadius, leakRate);
            obj.Reservoir2 = Reservoir(Nx, networkDensity, spectralRadius, leakRate);
            obj.Output = Output(Nx, Ny);
        end

        % Batch learning
        function WoutOpt = train(obj, UTrain, DTrain, optimizer)
            for i = 1:length(UTrain)
                % Reservoir1
                u = UTrain(i,:)';
                Win1_u = obj.Input1(u);
                x1 = obj.Reservoir1(Win1_u);

                % Reservoir2
                u = UTrain(i,:)';
                Win2_u = obj.Input2(u);
                x2 = obj.Reservoir1(Win2_u);

                x = [x1; x2];

                % Target data
                d = DTrain(i,:)';

                % Optimizer
                optimizer.update(x, d);
            end
            % Set optimal Wout.
            WoutOpt = optimizer.getWoutOpt();
            obj.Output.setWoutOpt(WoutOpt);
        end

        % Prediction after batch learning
        function YPred = predict(obj, UTest)
            YPred = zeros(length(UTest), obj.Ny);
            for i = 1:length(UTest)
                % Reservoir1
                u = UTest(i,:)';
                Win1_u = obj.Input1(u);
                x1 = obj.Reservoir1(Win1_u);

                % Reservoir2
                u = UTest(i,:)';
                Win2_u = obj.Input2(u);
                x2 = obj.Reservoir1(Win2_u);

                x = [x1; x2];

                % Model output after training
                yPred = obj.Output(x);
                YPred(i, :) = yPred';
            end
        end

        % Prediction after batch learning (free run)
        function YRun = run(obj, UTest)
            YRun = zeros(length(UTest), obj.Ny);
            u = UTest(1,:)';
            for i = 1:length(UTest)
                if (i > 1)
                    u(1:obj.Ny, :) = [];
                    u = [u; yRun'];
                end
                % Reservoir1
                Win1_u = obj.Input1(u);
                x1 = obj.Reservoir1(Win1_u);

                % Reservoir2
                Win2_u = obj.Input2(u);
                x2 = obj.Reservoir1(Win2_u);

                x = [x1; x2];

                % Model output after training
                yRun = obj.Output(x);
                YRun(i,:) = yRun';
            end
        end
    end
end
