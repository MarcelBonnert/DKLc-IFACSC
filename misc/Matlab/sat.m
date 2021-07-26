function u_out = sat(u,low,high)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

i_low = u < low;
i_high = u > high;

u_out = u;
u_out(i_low) = low;
u_out(i_high) = high;
% if u < low
%     u_out(i_low) = low;
% elseif u > high
%     u_out(i_high) = high;
% else
%     u_out = u;
% end
    
end

