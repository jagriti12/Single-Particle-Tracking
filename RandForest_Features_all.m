function [ftrs] = RandForest_Features(trjs)
    % Quick-call function to extract all features from a list of trajectories

    %% Initialize %%
    P = length(trjs);
    ftrs = zeros([6, P]);
    
    %% Evaluate %%
    wb = waitbar(0, sprintf('Extracting Features...'));


    for p = 1:length(trjs)        
        if(isempty(trjs{p})), continue; end
        
        try
            % Extract each feature in turn %
            ftrs(1,p) = Alpha(trjs{p});     % MSD based      
            ftrs (2, p) = Angular_Gaussianity(trjs{p});

            % % Debugging code to check eigenvalues
            [~, ~, lambda] = RadiusOfGyration(trjs{p}(:,1:3));
            if isnumeric(lambda) && numel(lambda) >= 3
                ftrs(3,p) = Asymmetry(trjs{p});     % Radius of Gyration based  %
            else
                % Handle cases where lambda doesn't have three elements
                fprintf('Warning: Eigenvalues do not have three elements.\n');
                ftrs(3,p) = NaN; % or handle it as needed
            end
            ftrs(4,p) = AvgMSDRatio(trjs{p});
            ftrs(5,p) = Efficiency(trjs{p});
            ftrs(6,p) = FractalDimension(trjs{p});  % Fractal based
            ftrs(7,p) = Gaussianity(trjs{p});

            jump_lengths = JumpLengths(trjs{p});
            ftrs(8, p) = mean(jump_lengths);
            ftrs(9,p) = Kurtosis(trjs{p});
            ftrs(10, p) = MaximalExcursion(trjs{p});
            ftrs (11, p)= MeanMaximalExcursion(trjs{p});
            ftrs(12,p) = Straightness(trjs{p});     % Linearity based           %
            ftrs(13,p) = Trappedness(trjs{p});
            ftrs(14, p) = VelocityAutocorrelationForN1(trjs{p});

        catch exception
            % Handle exceptions and errors gracefully
            fprintf('Error processing trajectory %d: %s\n', p, exception.message);
            ftrs(:, p) = NaN; % or handle it as needed
        end

        % Update the user
        if(mod(p,round(sqrt(P))) == 0)
            waitbar(p/P, wb, sprintf('Extracting Features... %5.2f%%', p/P*100));
        end
    end
    close(wb);
end

% 		% Extract each feature in turn %
% 		ftrs(1,p) = ExponentAlpha(trjs{p});		% MSD based					%
% 		ftrs(2,p) = MSDRatio(trjs{p});
% 		ftrs(3,p) = FractalDimension(trjs{p});% Fractal based%
%         % Debugging code to check eigenvalues
%         [T, V, lambda, R] = RadiusOfGyration(trjs{p});
%           if isnumeric(lambda) && numel(lambda) >= 3
%             ftrs(4,p) = Asymmetry(trjs{p});     % Radius of Gyration based  %
%           else
%                 % Handle cases where lambda doesn't have three elements
%              fprintf('Warning: Eigenvalues do not have three elements.\n');
%              ftrs(4,p) = NaN; % or handle it as needed
%            end
% 
% 
% % 		ftrs(4,p) = Asymmetry(trjs{p});			% Radius of Gyration based	%
% 		ftrs(5,p) = Straightness(trjs{p});		% Linearity based			%
% 		ftrs(6,p) = Trappedness(trjs{p});
% 		
% 		ftrs(7,p) = Gaussianity(trjs{p});
% 		ftrs(8,p) = Kurtosis(trjs{p});
% 		ftrs(9,p) = Efficiency(trjs{p});
% 		
% 		% Update the user %
% 		if(mod(p,round(sqrt(P))) == 0)
% 			waitbar(p/P, wb, sprintf('Extracting Features... %5.2f%%', p/P*100));
% 		end
% 	end
% 	close(wb);
% end
function [ftrs] = RandForest_Features_traj(trajs, wb, c, C)
	%% Argument Defaults %%
	if(nargin < 2), c = 0; C = 1; end

	%% Initialize %%
	[W, D, P] = size(trajs);	% Window length, Dimensions, Particles %
	ftrs = zeros([6, P]);

	%% Evaluate %%
	if(nargin < 3), wb = waitbar(0, 'Extracting features... (00.00%)'); end
	for p = 1:P
		% Temporary variable for convenience %
		traj = trajs(:,:,p);

		% Don't do anything if there's nothing to do %
%  		if(all(isnan(traj(:)))), continue; end

		% Only get the part that is not nan %
		traj = traj(~any(isnan(traj), 2), :);
%         traj = traj(~(isnan(traj{p,1})));
		% Extract each feature in turn %
		ftrs(1,p) = Alpha(traj);	% MSD based					%
		ftrs(2,p) = AvgMSDRatio(traj);
		ftrs(3,p) = FractalDimension(traj);	% Fractal based				%
		ftrs(4,p) = Asymmetry(traj);		% Radius of Gyration based	%
		ftrs(5,p) = Straightness(traj);		% Linearity based			%
		ftrs(6,p) = Trappedness(traj);
		ftrs(7,p) = Gaussianity(traj);
		ftrs(8,p) = Kurtosis(traj);
		ftrs(9,p) = Efficiency(traj);
		% Update the user %
		if(mod(p, round(P/10)) == 0)
			waitbar((c + p/P)/C, wb, sprintf('Extracting features... (%5.2f%%)', 100*(c + p/P)/C));
		end
	end

	%% Cleanup %%
	if(nargin < 2), close(wb); end
end




%% --- FEATURE DEFINITIONS --- %%
function [alpha] = Alpha(traj)	% Done	%
	% Derived from the MSD power-law fit %
	% ND ~ 1 | AD < 1 | CD < 1 | DM > 1 %
	
	%% Initialize %%
	N = size(traj, 1);			% Number of points		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	maxlag = round(sqrt(N));		% Max time lag %

	%% Evaluate %%
	% Evaluate the MSD %
	msd = MSD(pos, maxlag);%

    % Evaluate the average logarithmic derivative of the MSD (Ernst, et al.) %
	dmsd = diff(log( msd ));
	dt = diff(log( (1:maxlag)' ));%traj(2:maxlag+1, 4) ));
	
	%% Output %%
	alpha = mean(dmsd./dt);
end
function [msdr] = AvgMSDRatio(traj)		% Done	%
	% ND ~ 0 | AD > 0 | CD > 0 | DM < 0 % 
	
	%% Initialize %%
	N = size(traj, 1);			% Number of points		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	maxlag = round(N/3);%sqrt(N));		% Max time lag %
	
	%% Evaluate %%
	% Evaluate the MSD %
	msd = MSD(pos, maxlag);
	
	% Compare each possible time lag %
	msdr = nan([maxlag, maxlag]);
	for n1 = 1:maxlag-1
		for n2 = n1+1:maxlag
			msdr(n1,n2) = msd(n1)/msd(n2) - n1/n2;
		end
	end
	
	%% Output %%
	msdr = nanmean(msdr(:));
% 	msdr = min(max( msdr, -1/3),1);
end
function [g] = Gaussianity(traj)		% Done	%
	% Defined by Ernst, et al. %
    % Updated for 3D
	% ND ~ 0 | AD ! 0 | CD ! 0 | DM ! 0 %
	
	%% Initialize %%
	N = size(traj, 1) - 1;		% Number of steps		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	maxlag = round(sqrt(N));		% Max time lag %
	r2 = zeros([maxlag, 1]);
	r4 = zeros([maxlag, 1]);
	
	%% Evaluate %%
	% Evaluate the squared and quartic displacements %
	for n = 1:maxlag
		% Evaluate the squared displacements %
		sqdisp = sum( (pos(1+n:end,:) - pos(1:end-n,:)).^2, 2);
		
		% Evaluate the average squared & quartic displacements %
		r2(n) = mean(sqdisp);
		r4(n) = mean(sqdisp.^2);
	end

	%% Output %%
    % Updated for 3D
	g = mean(2*r4 ./ (3*r2.^2) - 1);
end

function [Df] = FractalDimension(traj)	% Done	%
	% Defined by Katz and George %
	%
	% Originally reported:
	%
	% However, one would imagine that Efficiency be between 0 and 1, meaning that a
	% perfectly efficient trajectory has an ending displacement equal to the sum of
	% the frame-to-frame displacements. Hence, we have omitted the (N-1) prefactor in
	% the denominator.
	%
	% ND ~ 2 | AD ~ 3 | CD ~ 3 | DM ~ 1 %
	
	%% Initialize %%
	N = size(traj, 1) - 1;		% Number of steps		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	%% Evaluate %%
	% Evaluate all frame-to-frame displacements %
	displacements = sqrt(sum((pos(2:end,:) - pos(1:end-1,:)).^2, 2));
	
	% Evaluate the largest distance between any two positions %
	d_max = 0;
	for i = 1:N
		dist = sqrt(max(sum((pos(i+1:end,:) - pos(i,:)).^2, 2)));
		if(dist > d_max), d_max = dist; end
	end
	
	% Evaluate the total length of the path %
	L = sum(displacements);
	
	%% Output %%
	Df = log(N)/log(N * d_max / L);
% 	Df = min(max( Df, 1), 4);
end
function [asym] = Asymmetry(traj)
    % To calculate asymmetry for 3D trajectories

    %% Initialize %%
    dims = size(traj, 2) - 1;    % Number of dimensions
    pos = traj(:, 1:dims);       % Get only the position values, exclude time

    %% Evaluate %%
    % Get the Radius of Gyration eigenvalues %
    [~, ~, lambda, ~] = RadiusOfGyration(pos);

    % Ensure lambda has three elements
    if numel(lambda) == 3
        % Shorthand
        lxy = (lambda(1) - lambda(2)) / (lambda(1) + lambda(2));
        lxz = (lambda(1) - lambda(3)) / (lambda(1) + lambda(3));
        lyz = (lambda(2) - lambda(3)) / (lambda(2) + lambda(3));

        %% Output %%
        asym = lxy^2 + lxz^2 + lyz^2;
    else
        % Handle the case when lambda does not have three elements
        asym = NaN;
        fprintf('Warning: Eigenvalues do not have three elements.\n');
        disp(lambda); % Debugging output
    end
end




% function [asym] = Asymmetry(traj)		% To 3D	%
% 	% Defined by Saxton and extended by Helmuth, et al. %
% 	% This was reported for 2D trajectories only, I'm not sure how to extend it to 3D
% 	% but I will do my best
% 	%
% 	% Originally reported:
% 	% a = -log( 1 - (l1 - l2)^2 / [2(l1 + l2)^2] )
% 	
% 	%% Initialize %%
% 	dims = size(traj, 2) - 1;	% Number of dimensions	%
% 	pos = traj(:,1:dims);		% Get only the position values, exclude time %
% 	
% 	%% Evaluate %%
% 	% Get the Radius of Gyration eigenvalues %
% 	[~, ~, l, ~] = RadiusOfGyration(pos);
% 	
% 	if(isempty(l))
% 		disp('ope');
% 	end
% 	% Shorthand %
% 	lxy = (l(1) - l(2))/(l(1) + l(2));
% 	lxz = (l(1) - l(3))/(l(1) + l(3));
% 	lyz = (l(2) - l(3))/(l(2) + l(3));
% 	
% 	%% Output %%
% 	asym = lxy^2 + lxz^2 + lyz^2;%-log(1 - num / (2 * den));
% end
function [k] = Kurtosis(traj)			% Done	%
	% Projects the positions onto the dominant eigenvector of the RoG tensor %

	%% Initialize %%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	%% Evaluate %%
	% Get the Radius of Gyration eigenvectors %
	[~, v, lambda, ~] = RadiusOfGyration(pos);
	
	% Project the positions onto the dominant eigenvector (1-D result --> dot prod) %
    [~,I] = max(abs(lambda));
    proj = pos * v(:,I);
	
	%% Output %%
	k = mean( (proj - mean(proj)).^4 / std(proj)^4 );
end

function [e] = Efficiency(traj)			% Done	%
	% Defined by Wagner, et al. %
	%
	% Originally reported:
	%
	
	%% Initialize %%
	N = size(traj, 1) - 1;			% Number of steps		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	%% Evaluate %%
	% Evaluate the ending squared displacement %
	disp_end = sum((pos(end,:) - pos(1,:)).^2, 2);
	
	% Evaluate the sum of frame-to-frame squared displacements %
	disp_f2f = sum( sum((pos(2:end,:) - pos(1:end-1,:)).^2, 2) );
	
	%% Output %%
	e = disp_end / (N * disp_f2f);
end
function [s] = Straightness(traj)		% Done	%
	% Similar to the Efficiency %
	
	%% Initialize %%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %
	
	%% Evaluate %%
	% Evaluate the ending squared displacement %
	disp_end = sqrt(sum((pos(end,:) - pos(1,:)).^2, 2));
	
	% Evaluate the sum of frame-to-frame squared displacements %
	disp_f2f = sum( sqrt(sum((pos(2:end,:) - pos(1:end-1,:)).^2, 2)) );
	
	%% Output %%
	s = disp_end / disp_f2f;
end
function [Pt] = Trappedness(traj)		% Done	%
	% Updated for 3D case
    % Defined by Saxton % 
	% Using only the first two time lags, estimate the short-time diffusion
	% coefficient D_short. Similarly, r0 is replaced by the estimated half of the
	% maximum distance between any two positions
	
	%% Initialize %%
	N = size(traj, 1) - 1;		% Number of steps		%
	dims = size(traj, 2) - 1;	% Number of dimensions	%
	pos = traj(:,1:dims);		% Get only the position values, exclude time %

	%% Evaluate %%
	% Evaluate the trapped time %
	t = traj(end,end) - traj(1,end);
	
	% Evaluate the maximum displacement %
	r0 = 0;
	for i = 1:N
		dist = sqrt(max(sum((pos(i+1:end,:) - pos(i,:)).^2, 2)));
        if dist > r0
            r0 = dist;
        end
    end 
%     r0 = r0;
% 	% Evaluate the short-time diffusion coefficient from the first two frames %
%     displacements_1 = mean(sqrt(sum((pos(2:end,:) - pos(1:end-1,:)).^2, 2)));
% 	displacements_2 = mean(sqrt(sum((pos(3:end,:) - pos(1:end-2,:)).^2, 2)));
% 	
%     D = (displacements_2 - displacements_1)/mean(diff(traj(:,end)));
%     if D <= 0
%         D = 1;
%     end

maxlag = round(sqrt(N+1));		% Max time lag %
% Evaluate the MSD %
msd = MSD(pos, maxlag);%
alpha = Alpha(traj);
tv = traj(2:maxlag+1,4);
Dv = msd./(tv.^alpha);
D = mean(Dv);



	%% Output %%
    Pt = 1-exp(0.731 -9.577*(D*t / r0^2 ));
%     Pt = 1 - exp(0.731 -9.577*(D*t / r0^2 ));
end
function [maxExcursion] = MaximalExcursion(traj)
    % Calculate Maximal Excursion
    positions = traj(:, 1:end-1); % Extract positions, excluding time
    displacements = diff(positions); % Calculate displacements between consecutive positions
    distances = sqrt(sum(displacements.^2, 2)); % Calculate distances traveled between consecutive points
    maxdist = max(distances); % Find the maximum distance
    maxExcursion = maxdist./sqrt(sum((positions(end,:)-positions(1,:)).^2, 2));
end
function vacf = VelocityAutocorrelationForN1(traj)
    % Calculate the velocity autocorrelation function for lag 1 (n=1)
    N = size(traj, 1);
    pos = traj(:, 1:end-1); % Extract positions, excluding time
    vacf = 0;
    for i = 1:(N - 2) % Adjusted to N-2 to accommodate i+1+n in the loop
        deltaV1 = sqrt(sum((pos(i + 2, :) - pos(i + 1, :)).^2,2)); % Distance (~Velocity) at i+n+1
        deltaV2 = sqrt(sum((pos(i + 1, :) - pos(i, :)).^2,2));     % Distance (~Velocity) at i
        vacf = vacf + deltaV1.*deltaV2;
    end
    vacf = vacf / (N - 2); % Normalizing by N - 2
 end


function [jump_lengths] = JumpLengths(traj)
    % Calculate the jump lengths of the trajectory
    
    % Extract positions from the trajectory
    positions = traj(:, 1:end-1); % Exclude the time column
    
    % Initialize an array to store jump lengths
    jump_lengths = zeros(size(positions, 1) - 1, 1);
    
    % Calculate jump lengths for each step in the trajectory
    for i = 1:size(positions, 1) - 1
        jump_lengths(i) = norm(positions(i+1, :) - positions(i, :));
    end
end
function [mean_excursion] = MeanMaximalExcursion(traj)
    % Calculate the mean maximal excursion of the trajectory
    
    % Extract positions from the trajectory
    positions = traj(:, 1:end-1); % Exclude the time column
    
     % Calculate the maximum excursion (max distance traveled)
    max_excursion = max(vecnorm((positions - positions(1,:)), 2, 2));
    
    % Calculate the total time duration
    t0 = traj(1, end);
    tN = traj(end, end);
    
    % Calculate the number of data points
    N = size(positions, 1);
    
    % Calculate the standard deviation estimator
    sigma_hat_sq = 0;
    for j = 2:N
        sigma_hat_sq = sigma_hat_sq + norm(positions(j, :) - positions(j-1, :))^2;
    end
    sigma_hat_sq = sigma_hat_sq / (2 * N * (tN - t0));
    
    % Calculate the mean maximal excursion
    mean_excursion = max_excursion / (sqrt(2 * sigma_hat_sq * (tN - t0)));
end
function [g, hgi] = Angular_Gaussianity(traj)
    % Calculate Gaussianity features of a 3D trajectory
    % Number of steps
    N = size(traj,1) - 1;
 
    % Calculate the displacement vectors
    dx = diff(traj(:, 1));
    dy = diff(traj(:, 2));
    dz = diff(traj(:, 3));
    v_norm = sqrt(dx.^2 + dy.^2 + dz.^2);
    % Main vector
    mv = traj(end,1:3) - traj(1,1:3);
    mv_norm = norm(mv);
    % Calculate the angle (radians) between main and current vector
    theta = zeros(1, N);
    for i = 1:N
        dotProduct = dot(mv, [dx(i) dy(i) dz(i)]);
        theta(i) = acos(dotProduct./ (mv_norm.*v_norm(i)));    
    end
    dAngle = diff(theta);

 
    % Calculate the histogram of angular displacement changes
    edges = linspace(-pi, pi, 50);  % Define histogram bins for combined angular change
    counts = histcounts(dAngle, edges);
 
    % Calculate Gaussianity (standard deviation) of the histogram
    sigma = std(counts);
 
    % Calculate higher-order Gaussianity index (HGI)
    n = numel(counts);
    hgi = 0;
    for i = 1:n
        hgi = hgi + (counts(i) / n - 1/n)^4;
    end
 
    % Normalize HGI
    hgi = hgi / (n^2 * (1/n - 1));
 
    % Return Gaussianity and HGI as features
    g = sigma;
end


% function [gamma_p, P] = EmpiricalPVariationFeatures(traj)
%     % Calculate six features based on empirical p-variation
% 
%     N = length(traj);
%     
%     % Initialize variables to store the results
%     gamma_p = zeros(1, 5);  % Features 1 to 5
%     P = 0;  % Feature 6
% 
%     % Loop over lags from 1 to 5
%     for lag = 1:5
%         V_m = 0;
%         for k = 1:N-lag
%             V_m = V_m + abs(traj(k+lag) - traj(k))^lag;
%         end
%         % Add a small epsilon to avoid division by zero
%         V_m = V_m / (N - lag + eps);
% 
%         % Calculate the power gamma_p for each lag
%         if V_m > 0
%             gamma_p(lag) = log(V_m) / log(lag);
%         else
%             gamma_p(lag) = 0; % Handle the case where V_m is zero
%         end
%     end
% 
%     % Find the highest p for which V(p)_m is not monotonous
%     max_p = 0;
%     max_P = 0;
%     V_p_m_prev = 0;
%     for p = 1:5
%         V_p_m = 0;
%         for k = 1:N-1
%             V_p_m = V_p_m + abs(traj(k+1) - traj(k))^p;
%         end
%         % Add a small epsilon to avoid division by zero
%         V_p_m = V_p_m / (N - 1 + eps);
% 
%         % Check monotonicity
%         if p > 1 && V_p_m < V_p_m_prev
%             % Monotonicity changed
%             if V_p_m > V_p_m_prev
%                 max_P = 1;  % Convex
%             else
%                 max_P = -1;  % Concave
%             end
%             max_p = p - 1;
%             break;
%         end
%         V_p_m_prev = V_p_m;
%     end
% 
%     % Format the output as real numbers with two decimal places
%     gamma_p = round(gamma_p(1:max_p), 2);
%     P = round(max_P, 2);
% end


%% --- HELPER FUNCTIONS --- %%
function [msd] = MSD(traj, maxlag)
	% Evaluates the MSD of a trajectory for all possible time-lags %
	
	%% Initialize %%
	N = size(traj, 1);
	dims = size(traj, 2);
	
	if(nargin < 2), maxlag = N-1; end
	msd = zeros([maxlag, 1]);
	
	%% Evaluate %%
	pos = traj(:,1:dims);
	for n = 1:maxlag
		msd(n) = mean(sum((pos(1+n:end,:)-pos(1:end-n,:)).^2, 2), 1);
	end
	
	%% Output %%
	% msd %
end
function [T, V, lambda, R] = RadiusOfGyration(traj)
    % Evaluates the Radius of Gyration tensor (T), eigenvectors (V), eigenvalues
    % (lambda), and Radius of Gyration (R) of a trajectory. 
    % Assumes traj is a 2D array of positions accessed via: traj(frame, dim).
    % Time is never included as the last dimension (e.g., [x,y], [x,y,z]).

    %% Initialization %%
    dims = size(traj, 2);        % Number of dimensions
    T = zeros([dims, dims]);     % Radius of Gyration tensor

    %% Evaluate %%
    traj_ = traj - mean(traj, 1);    % Offset each dimension by the mean

    % Calculate each element of the RoG tensor
    for i = 1:dims
        for j = i:dims
            T(i,j) = mean( traj_(:,i) .* traj_(:,j), 1 );
            T(j,i) = T(i,j); % Symmetric tensor
        end
    end

    % Calculate the eigenvectors and eigenvalues
    if(nargout > 1)
        [V, D] = eig(T);    % Remember T * V = V * D
        lambda = diag(D);   % Get the diagonal elements (eigenvalues)
        lambda = real(lambda); % Ensure lambda is real to avoid complex values
        
        % Ensure that lambda always has three elements by padding with zeros if needed
        lambda = [lambda; zeros(3 - length(lambda), 1)];
    end

    % Calculate the radius of gyration
    if(nargout > 3)
        R = sqrt(sum(lambda));
    end
end




