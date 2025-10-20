/**
 * PRELOAD
 * 
 * loading will be end after document is loaded
 */

const preloader = document.querySelector("[data-preaload]");

window.addEventListener("load", function () {
  preloader.classList.add("loaded");
  document.body.classList.add("loaded");
});

// Add this script to hide the preloader after the page loads
window.addEventListener("load", () => {
  const preloader = document.querySelector(".preload");
  if (preloader) {
    preloader.style.display = "none";
  }
});

/**
 * add event listener on multiple elements
 */
  
  // Navbar
  
  document.addEventListener('click', e => {
    const menu = document.getElementById('mobileMenu');
    if (e.target.closest('#menuBtn')) menu.classList.toggle('hidden');
    else if (!e.target.closest('nav')) menu.classList.add('hidden');
});
window.addEventListener('resize', () => {
    if (window.innerWidth >= 768) document.getElementById('mobileMenu').classList.add('hidden');
});
  
  
  //  carousel
  
   // For carousel
   const carousel = document.getElementById('carousel').querySelector('div');
   const slides = carousel.children;
   const prevButton = document.getElementById('prev');
   const nextButton = document.getElementById('next');
   let carouselIndex = 0; // Renamed variable
  
   function updateCarousel() {
       const offset = -carouselIndex * slides[0].offsetWidth;
       carousel.style.transform = `translateX(${offset}px)`;
   }
  
   prevButton.addEventListener('click', () => {
       carouselIndex = (carouselIndex - 1 + slides.length) % slides.length;
       updateCarousel();
   });
  
   nextButton.addEventListener('click', () => {
       carouselIndex = (carouselIndex + 1) % slides.length;
       updateCarousel();
   });
  
   // Auto-slide every 5 seconds
   setInterval(() => {
       carouselIndex = (carouselIndex + 1) % slides.length;
       updateCarousel();
   }, 5000);