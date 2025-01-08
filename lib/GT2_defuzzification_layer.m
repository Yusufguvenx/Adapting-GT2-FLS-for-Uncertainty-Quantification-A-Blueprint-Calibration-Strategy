function [output_lower, output_upper, output_mean] = GT2_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u)
% v0.2 compatible with minibatch
%
%
% calculating the weighted sum with firts calcutating the weighted
% elements then adding them
%
% @param output -> output
%
%       (1,1,mbs) tensor
%       mbs = mini-batch size
%       (:,:,1) -> defuzzified output of the first element of the batch
%
% @param input 1 -> normalized_firing_strength
%
%      (rc,1,mbs) tensor
%       rc = number of rules
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input 2 -> output_mf
%
%       (rc,1) vector
%       rc = number of rules
%       (1,1) -> constant or value of the first output membership function

if output_type == "singleton"
    if type_reduction_method == "KM"

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        den2 = delta_f*u;

        num2 = (permute(learnable_parameters.pmf.singleton.c,[3 1 2]).*delta_f)*u;
        num2 = permute(num2,[3,2,1]);
        num1 = sum(learnable_parameters.pmf.singleton.c.* lower_firing_strength,1);

        num = num1 + num2;
        den2 = permute(den2,[3,2,1]);
        den1 = sum(lower_firing_strength,1);

        den = den1 + den2;


        output = num./den;
        output_lower = min(output,[],2);
        output_upper = max(output,[],2);

        output_mean = (output_lower + output_upper)./2;
    end

elseif output_type == "linear"

    if type_reduction_method == "KM"

        temp_mf = [learnable_parameters.pmf.linear.a,learnable_parameters.pmf.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

%         c = permute(c, [1, 3, 2]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        den2 = pagemtimes(delta_f, u);
        
        num2 = permute(c, [3, 1, 2]).*delta_f;
        num2 = pagemtimes(num2, u);
        num2 = permute(num2,[3,2,1]);

        num1 = sum(c .* lower_firing_strength,1);
        num1 = permute(num1, [2, 1, 3]);

        num = num1 + num2;

        den2 = permute(den2,[3,2,1]);
        den1 = sum(lower_firing_strength,1);
        den1 = permute(den1, [2, 1, 3]);

        den = den1 + den2;

        output = num./den;

        output_lower = min(output,[],2);
        output_upper = max(output,[],2);
        output_mean = (output_lower + output_upper)./2;
    end

end

output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end