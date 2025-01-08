function alpha_opt = derivative_free_search_algorithm(x, y, number_mf, number_inputs, number_outputs, mbs, Learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha0, delta, alpha_rev, step_size, exploration_factor, tolerance, target)
    % derivative_free_search_algorithm with alpha bounded between [0.01, 1]
    % Inputs:
    % f - function handle (e.g., @(alpha) alpha^2 + 3*alpha + 7)
    % alpha0 - initial guess for alpha
    % step_size - initial step size
    % exploration_factor - factor by which the step size is reduced
    % tolerance - tolerance level for convergence
    % target - target value for f(alpha) (e.g., 95)
    
    % Ensure alpha0 is within bounds
    alpha_current = max(min(alpha0, 1), 0.01);
    step = step_size;

    while 1
        % 1. Exploratory search
        [y_l, y_u] = GT2_fismodel_LA1_per_alpha(x, number_mf, number_inputs, number_outputs, mbs, Learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha_current, delta, alpha_rev);
        
        y_u = permute(y_u, [1 3 2]);
        y_l = permute(y_l, [1 3 2]);
        
        PICP_current = PICP(y, y_l, y_u);

        % Move forward and backward with boundary checks
        alpha_plus = min(alpha_current + step, 1);
        alpha_minus = max(alpha_current - step, 0.01);
        
        [y_l_plus, y_u_plus] = GT2_fismodel_LA1_per_alpha(x, number_mf, number_inputs, number_outputs, mbs, Learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha_plus, delta, alpha_rev);

        y_u_plus = permute(y_u_plus, [1 3 2]);
        y_l_plus = permute(y_l_plus, [1 3 2]);

        [y_l_minus, y_u_minus] = GT2_fismodel_LA1_per_alpha(x, number_mf, number_inputs, number_outputs, mbs, Learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha_minus, delta, alpha_rev);

        y_u_minus = permute(y_u_minus, [1 3 2]);
        y_l_minus = permute(y_l_minus, [1 3 2]);     
        
        PICP_plus = PICP(y, y_l_plus, y_u_plus);
        PICP_minus = PICP(y, y_l_minus, y_u_minus);

        PICP_plus = gather(extractdata(PICP_plus));
        PICP_minus = gather(extractdata(PICP_minus));
        PICP_current = gather(extractdata(PICP_current));


       if norm(PICP_plus - target, 1) < norm(PICP_current - target, 1) && norm(PICP_minus - target, 1) < norm(PICP_current - target, 1)
            % Both directions are improvements, pick the closer one
            if norm(PICP_plus - target, 1) < norm(PICP_minus - target, 1)
                alpha_next = alpha_plus;
            else
                alpha_next = alpha_minus;
            end
        elseif norm(PICP_plus - target, 1) < norm(PICP_current - target, 1)
            alpha_next = alpha_plus;
        elseif norm(PICP_minus - target, 1) < norm(PICP_current - target, 1)
            alpha_next = alpha_minus;
        elseif step < 1e-4
            break;
        else
            % If neither direction improves, reduce the step size
            step = step * exploration_factor;
            continue;
        end
        % 2. Pattern move
        alpha_current = alpha_next;

        [y_l, y_u] = GT2_fismodel_LA1_per_alpha(x, number_mf, number_inputs, number_outputs, mbs, Learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha_current, delta, alpha_rev);
        
        y_u = permute(y_u, [1 3 2]);
        y_l = permute(y_l, [1 3 2]);
        
        PICP_current = PICP(y, y_l, y_u);
        PICP_current = gather(extractdata(PICP_current));

        % 3. Check for convergence
        if norm(PICP_current - target, 1) < tolerance
            break;
        end
    end
    
    alpha_opt = alpha_current;  % Return the optimal alpha found
end

