from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point
import geopandas as gpd


def add_scalebar(ax, fontsize=20, location='lower left'):
    """
    Adds a scalebar to a map plot. Assumes that the map is plotted using
    WGS 84 - degrees and that it is centered around Europe.

    Adapten from:
    https://geopandas.org/en/stable/gallery/matplotlib_scalebar.html#Geographic-coordinate-system-(degrees)
    """

    # Two points roughly at Berlin coordinates:
    # more or less the center of Europe

    lon, lat = (13.4, 52.5)
    points = gpd.GeoSeries(
        [Point(lon, lat), Point(lon - 1, lat)], crs=4326
    )  # Geographic WGS 84 - degrees
    points = points.to_crs(32619)  # Projected WGS 84 - meters

    distance_meters = points[0].distance(points[1])

    scalebar = ScaleBar(
        distance_meters,
        location=location,
        font_properties={"size": fontsize}
    )

    ax.add_artist(scalebar)


def add_north_arrow(
        ax,
        fontsize=20,
        x=0.2,
        y=0.1,
        arrow_length=0.07,
        width=5,
        headwidth=15,
        headlength=None
):
    x, y, arrow_length = x, y, arrow_length
    if headlength is None:
        headlength = 1.5 * headwidth
    ax.annotate(
        'N',
        xy=(x, y),
        xytext=(x, y-arrow_length),
        arrowprops=dict(
            facecolor='black',
            width=width,
            headwidth=headwidth,
            headlength=headlength
        ),
        ha='center',
        va='center',
        fontsize=fontsize,
        xycoords=ax.transAxes
    )
