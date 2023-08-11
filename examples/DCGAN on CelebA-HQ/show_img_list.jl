using Plots, Measures, Images, FileIO

# plot all batches in img_list by arranging the images in a batch in a grid
# press enter in the console to continue
function show_img_list(img_list)
    for (i, img_batch) in enumerate(img_list)
        batch_size = size(img_batch)[end]

        image_plots = []
        for index_batch in 1:batch_size
            image = @view img_batch[:, :, :, index_batch]
            image = PermutedDimsArray(image, (3, 2, 1))

            # normalize
            min = minimum(image)
            max = maximum(image)
            norm(x) = (x - min) / (max - min)
            image = norm.(image)

            image = colorview(RGB, image)
            image_plot = plot(image)
            push!(image_plots, image_plot)
        end 

        # create a plot and display a gui window with the plot
        p = plot(image_plots..., framestyle=:none, border=:none, leg=false, ticks=nothing, margin=-1.5mm, left_margin=-1mm, right_margin=-1mm) # , show=true
        display(p)
        # prevent the window from closing immediately
        readline()
        # save the plot as an image file 
        savefig(p, "img_list_grid_$i.png")

        println("[$i/$(length(img_list))]")
    end
end

file_name_img_list = "img_list.jld2"
img_list = load(file_name_img_list, "img_list")
println(length(img_list))
img_list = img_list[end-9:end] # show only the last 10 batches
show_img_list(img_list)