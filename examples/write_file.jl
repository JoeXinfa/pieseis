# Write JavaSeis with Julia package TeaSeis.jl

# Run from Julia REPL
# julia> include("C:/Users/xinfa/Documents/code/pieseis/examples/write_file.jl")
# RUn from command line (Linux terminal or Windows command prompt)
# C:\Users\xinfa\AppData\Local\Julia-1.0.1\bin\julia.exe C:\Users\xinfa\Documents\code\pieseis\examples\write_file.jl

println("start")

push!(LOAD_PATH, "C:/Users/xinfa/Documents/code/TeaSeis.jl/src")
using TeaSeis
println("done load")

filename = "C:/Users/xinfa/Documents/181116_julia.js"
io = jsopen(filename, "w", axis_lengths=[101, 101, 101], dataformat=Int16)
println("done open")

trcs, hdrs = allocframe(io)

map(i->set!(prop(io, stockprop[:TRC_TYPE]), hdrs, i, tracetype[:live]), 1:size(io,2))
map(i->set!(prop(io, stockprop[:TRACE]   ), hdrs, i, i               ), 1:size(io,2))
map(i->set!(prop(io, stockprop[:FRAME]   ), hdrs, i, 1               ), 1:size(io,2))
println("done write headers")

using Random
rand!(trcs)
writeframe(io, trcs, hdrs)
println("done write traces")

close(io)
