#+ warning=FALSE, message=FALSE
library(keras)
K <- backend()

model <- application_vgg16(weights = "imagenet") 
model  
#' ![vgg16](C:/Users/hyun3/Desktop/jm_paper/vgg16.png)

#' ## Elephant Example
#+ fig.align = 'center', warning=FALSE, message=FALSE
image <- get_file("elephant.jpg", "https://goo.gl/zCTWXW") %>% 
  image_load(target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(image)
imagenet_decode_predictions(preds, top = 3)[[1]]

which.max(preds[1,])


african_elephant_output <- model$output[, 387]
african_elephant_output
last_conv_layer <- model %>% get_layer("block5_conv3")
last_conv_layer
grads <- K$gradients(african_elephant_output, last_conv_layer$output)[[1]]
grads
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
pooled_grads
iterate <- K$`function`(list(model$input), 
                        list(pooled_grads, last_conv_layer$output[1,,,])) 

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(image))

for (i in 1:512) {
    conv_layer_output_value[,,i] <- 
      conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
dim(conv_layer_output_value)
dim(pooled_grads_value)


heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)
dim(heatmap)
heatmap
  
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                            bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/elephant_heatmap.png") 

library(magick) 
library(viridis) 

img_path <- 'C:/Users/hyun3/Documents/.keras/datasets/elephant.jpg'
image <- image_read(img_path) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/elephant_overlay.png", 
#               width = 14, height = 14, bg = NA, col = pal_col) 

image_read("C:/Users/hyun3/Desktop/jm_paper/elephant_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()
#'

#' ### Puppy Example
#+ fig.align = 'center', warning=FALSE, message=FALSE
image <- get_file("puppy.png", 
                  "https://www.guidedogsvictoria.com.au/wp-content/themes/default/static/img/puppy.png") %>% 
  image_load(target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(image)
imagenet_decode_predictions(preds, top = 3)[[1]]

which.max(preds[1,])


Labrador_retriever_output <- model$output[, 209]
last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- K$gradients(Labrador_retriever_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
iterate <- K$`function`(list(model$input), 
                        list(pooled_grads, last_conv_layer$output[1,,,])) 

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(image))

for (i in 1:512) {
    conv_layer_output_value[,,i] <- 
      conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/Labrador_retriever_heatmap.png") 

img_path <- 'C:/Users/hyun3/Documents/.keras/datasets/puppy.png'
image <- image_read(img_path) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/puppy_overlay.png", 
#               width = 14, height = 14, bg = NA, col = pal_col) 

image_read("C:/Users/hyun3/Desktop/jm_paper/puppy_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()
#'



#' ### Puppy Example 2  
#+ fig.align = 'center', warning=FALSE, message=FALSE
image <- get_file("dog-labrador-puppy.jpg", 
                  "https://s3.us-west-2.amazonaws.com/nextpaw.com/cms/images/BackgroundImage/1470/dog-labrador-puppy.jpg") %>% 
  image_load(target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(image)
imagenet_decode_predictions(preds, top = 3)[[1]]

which.max(preds[1,])


puppy2_output <- model$output[, 208]
last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- K$gradients(puppy2_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
iterate <- K$`function`(list(model$input), 
                        list(pooled_grads, last_conv_layer$output[1,,,])) 

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(image))

for (i in 1:512) {
    conv_layer_output_value[,,i] <- 
      conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/puppy2_heatmap.png") 

img_path <- 'C:/Users/hyun3/Documents/.keras/datasets/dog-labrador-puppy.jpg'
image <- image_read(img_path) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/puppy2_overlay.png", 
#               width = 14, height = 14, bg = NA, col = pal_col) 

image_read("C:/Users/hyun3/Desktop/jm_paper/puppy2_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()
#'


#' ### Flower Example
#+ fig.align = 'center', warning=FALSE, message=FALSE
image <- get_file("012.jpg", 
                  "http://www.seoulmilkblog.co.kr/wp/wp-content/uploads/2018/03/012.jpg") %>% 
  image_load(target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(image)
imagenet_decode_predictions(preds, top = 3)[[1]]

which.max(preds[1,])


flower_output <- model$output[, 310]
last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- K$gradients(flower_output, last_conv_layer$output)[[1]]
pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
iterate <- K$`function`(list(model$input), 
                        list(pooled_grads, last_conv_layer$output[1,,,])) 

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(image))

for (i in 1:512) {
    conv_layer_output_value[,,i] <- 
      conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/flower_heatmap.png") 

img_path <- 'C:/Users/hyun3/Documents/.keras/datasets/012.jpg'
image <- image_read(img_path) 
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
# write_heatmap(heatmap, "C:/Users/hyun3/Desktop/jm_paper/flower_overlay.png", 
#               width = 14, height = 14, bg = NA, col = pal_col) 

image_read("C:/Users/hyun3/Desktop/jm_paper/flower_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()
#'
