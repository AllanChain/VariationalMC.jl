using Dates

mutable struct StatsLogger
    stream::IOStream
    should_add_header::Bool
end


function with_stats(
    f::Function,
    restore_path::AbstractString,
    save_path::AbstractString;
    optimizing::Bool = true,
)
    if save_path == ""
        f(nothing)
    end
    stats_filename = optimizing ? "optim-stats.csv" : "eval-stats.csv"
    stats_file = joinpath(save_path, stats_filename)
    if restore_path != save_path && ispath(stats_file)
        error("Cannot overwrite stats file $stats_file without restoring from it!")
    end
    if restore_path == "" # Nothing to restore
        fresh_start = true
    elseif restore_path != save_path # Restoring from different location
        prev_stats_file = joinpath(restore_path, stats_filename)
        if filesize(prev_stats_file) > 0 # File exists and has content
            cp(prev_stats_file, stats_file)
            fresh_start = false
        else
            fresh_start = true
        end
    elseif filesize(stats_file) > 0 # Restoring from same location, has content
        fresh_start = false
    else
        fresh_start = true
    end
    mode = fresh_start ? "w" : "a"
    stats = StatsLogger(open(stats_file, mode), fresh_start)
    f(stats)
    close(stats.stream)
end

function log_stats(
    stats::Union{StatsLogger,Nothing},
    saving_data::AbstractDict,
    printing_data::AbstractDict,
)
    if stats !== nothing
        if stats.should_add_header
            write(stats.stream, join(keys(saving_data), ",") * "\n")
            stats.should_add_header = false
        end
        write(stats.stream, join(values(saving_data), ",") * "\n")
        flush(stats.stream)
    end

    print(Dates.format(now(), "[yyyy-mm-dd HH:MM:SS.sss] "))
    for data in (saving_data, printing_data)
        for (k, v) in data
            print("$k: $v; ")
        end
    end
    println()
end
