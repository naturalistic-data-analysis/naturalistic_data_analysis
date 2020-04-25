# Speakers
[Chris Baldassano](http://www.dpmlab.org/) Columbia University

[Linda Geerligs](https://www.ru.nl/english/people/geerligs-l/) Donders Institute for Brain, Cognition, and Behavior

[Juha Lahnakoski](https://users.aalto.fi/~jlahnako/) Forschungszentrum Jülich

[Carolyn Parkinson](http://csnlab.org/) University of California Los Angeles

[Won Mok Shim](http://wshimlab.com/) SungKyunKwan University

[Tal Yarkoni](https://talyarkoni.org/) University of Texas at Austin

[Yaara Yeshurun](https://people.socsci.tau.ac.il/mu/yaarayeshurun/) Tel-Aviv University

<!-- Using Cards -->
<section>
  <h2>Speakers</h2>
  <!-- <div class="container" id="faculty"> -->
    <div class="row" id="faculty">
      {% for person in site.data.speakers %}
          <div class="col s12 m6 l4">
            <div class="card hoverable" id="faculty">
              <div class="card-image" id="faculty">
                <a href="{{person.Website}}"><img src="{{site.url}}/images/speakers/{{person.Picture}}"></a>
              </div>
              <div class="card-content">
                <span class="card-title center"><a href="{{person.Website}}">{{person.First}} <span>{{person.Last}}</span></a></span>
                <p class="center card-affiliation">{{person.Institution}}</p>
              </div>
            </div>
          </div>
      {% endfor %}
    </div>
  <!-- </div> -->
</section>

