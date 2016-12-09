var data = d3.csv("vgsales.csv", function(d) {
  return {
    EU_Sales : +d.EU_Sales,
    Genre : d.Genre,
    Global_Sales : +d.Global_Sales,
    JP_Sales : +d.JP_Sales,
    NA_Sales : +d.NA_Sales,
    Name : d.Name,
    Other_Sales : +d.Other_Sales,
    Platform : d.Platform,
    Publisher : d.Publisher,
    Rank : +d.Rank,
    Year : +format.parse(d['date'])
  };
});
