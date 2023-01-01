using Dates

mutable struct StatsLogger
    stream::IOStream
    should_add_header::Bool
end

const STATS_FILE = "vmcjl-stats.csv"


function with_stats(f::Function, restore_path::AbstractString, save_path::AbstractString)
    stats_file = joinpath(save_path, STATS_FILE)
    if restore_path != save_path && ispath(stats_file)
        error("Cannot overwrite stats file $stats_file without restoring from it!")
    end
    if restore_path == "" # Nothing to restore
        fresh_start = true
    elseif restore_path != save_path # Restoring from different location
        prev_stats_file = joinpath(restore_path, STATS_FILE)
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

function log_stats(stats::StatsLogger, d::AbstractDict)
    if stats.should_add_header
        write(stats.stream, join(keys(d), ",") * "\n")
        stats.should_add_header = false
    end
    write(stats.stream, join(values(d), ",") * "\n")
    print(Dates.format(now(), "[yyyy-mm-dd HH:MM:SS.sss] "))
    for (k, v) in d
        print("$k: $v; ")
    end
    println()
end
